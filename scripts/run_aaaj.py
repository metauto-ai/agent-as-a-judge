import re
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

from agent_as_a_judge.agent import JudgeAgent
from agent_as_a_judge.config import AgentConfig


def main(agent_config: AgentConfig, logger: logging.Logger):

    def extract_number_from_filename(filename: str) -> int:
        match = re.search(r"(\d+)", filename)
        return int(match.group(1)) if match else float("inf")

    instance_files = sorted(
        list(agent_config.instance_dir.glob("*.json")),
        key=lambda f: extract_number_from_filename(f.stem),
    )

    logger.info(f"Total instances found: {len(instance_files)}")

    for instance_file in instance_files:
        instance_name = instance_file.stem

        trajectory_file = None
        if agent_config.trajectory_file:
            trajectory_file = agent_config.trajectory_file / f"{instance_name}.json"

        judgment_file = agent_config.judge_dir / instance_file.name

        if judgment_file.exists():
            logger.info(
                f"Judgment for instance '{instance_name}' already exists. Skipping..."
            )
            continue

        if trajectory_file and trajectory_file.exists():
            logger.info(
                f"Processing instance: {instance_file} with trajectory: {trajectory_file}"
            )
        else:
            logger.warning(
                f"Trajectory file not found for instance: {instance_file}, processing without it"
            )
            trajectory_file = None

        workspace = agent_config.workspace_dir / instance_name

        judge_agent = JudgeAgent(
            workspace=workspace,
            instance=instance_file,
            judge_dir=agent_config.judge_dir,
            trajectory_file=trajectory_file,
            config=agent_config,
        )
        judge_agent.judge_anything()


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--developer_agent", type=str, required=True, help="Name of the developer agent"
    )
    parser.add_argument(
        "--setting",
        type=str,
        required=True,
        help="Setting for the JudgeAgent (e.g., gray_box, black_box)",
    )
    parser.add_argument(
        "--planning",
        type=str,
        required=True,
        choices=["planning", "comprehensive (no planning)", "efficient (no planning)"],
        help="Module to run",
    )
    parser.add_argument(
        "--benchmark_dir",
        type=str,
        required=True,
        help="Base directory for the DevAI benchmark",
    )
    parser.add_argument(
        "--include_dirs",
        nargs="+",
        default=["src", "results", "models", "data"],
        help="Directories to include in search",
    )
    parser.add_argument(
        "--exclude_dirs",
        nargs="+",
        default=[
            "__pycache__",
            "env",
            ".git",
            "venv",
            "logs",
            "output",
            "tmp",
            "temp",
            "cache",
            "data",
        ],
        help="Directories to exclude in search",
    )
    parser.add_argument(
        "--exclude_files",
        nargs="+",
        default=[".DS_Store"],
        help="Files to exclude in search",
    )
    parser.add_argument(
        "--trajectory_file",
        type=str,
        help="Path to the trajectory directory, if available",
    )
    parser.add_argument(
        "--majority_vote",
        type=int,
        default=1,
        help="Number of independent judge calls per requirement for confidence estimation",
    )
    parser.add_argument(
        "--critical_threshold",
        type=float,
        default=0.5,
        help="Decision threshold over satisfied vote ratio",
    )

    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()

    benchmark_dir = Path(args.benchmark_dir)
    instance_dir = benchmark_dir / "devai/instances"
    workspace_dir = benchmark_dir / f"workspaces/{args.developer_agent}"
    judge_dir = (
        benchmark_dir
        / f"judgment/{args.developer_agent}/agent_as_a_judge/{args.setting}"
    )
    trajectory_file = benchmark_dir / f"trajectories/{args.developer_agent}"

    agent_config = AgentConfig(
        include_dirs=args.include_dirs,
        exclude_dirs=args.exclude_dirs,
        exclude_files=args.exclude_files,
        setting=args.setting,
        planning=args.planning,
        judge_dir=judge_dir,
        workspace_dir=workspace_dir,
        instance_dir=instance_dir,
        trajectory_file=trajectory_file,
        majority_vote=args.majority_vote,
        critical_threshold=args.critical_threshold,
    )

    main(
        agent_config=agent_config,
        logger=logger,
    )
