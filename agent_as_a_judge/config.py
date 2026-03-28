from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass
class AgentConfig:
    include_dirs: Optional[List[str]] = None
    exclude_dirs: Optional[List[str]] = None
    exclude_files: Optional[List[str]] = None
    setting: str = "gray_box"
    planning: str = "efficient (no planning)"
    judge_dir: Optional[Path] = None
    workspace_dir: Optional[Path] = None
    instance_dir: Optional[Path] = None
    trajectory_file: Optional[Path] = None
    majority_vote: int = 1
    critical_threshold: float = 0.5

    @classmethod
    def from_args(cls, args):

        return cls(
            include_dirs=(
                args.include_dirs
                if hasattr(args, "include_dirs")
                else ["src", "results", "models", "data"]
            ),
            exclude_dirs=(
                args.exclude_dirs
                if hasattr(args, "exclude_dirs")
                else ["__pycache__", "env"]
            ),
            exclude_files=(
                args.exclude_files if hasattr(args, "exclude_files") else [".DS_Store"]
            ),
            setting=args.setting,
            planning=args.planning,
            judge_dir=Path(args.judge_dir),
            workspace_dir=Path(args.workspace_dir),
            instance_dir=Path(args.instance_dir),
            trajectory_file=(
                Path(args.trajectory_file) if args.trajectory_file else None
            ),
            majority_vote=(
                args.majority_vote if hasattr(args, "majority_vote") else 1
            ),
            critical_threshold=(
                args.critical_threshold
                if hasattr(args, "critical_threshold")
                else 0.5
            ),
        )
