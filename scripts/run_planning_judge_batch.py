#!/usr/bin/env python3
"""
Batch script to run Planning Judge against multiple DevAI instances.

This script allows you to evaluate your planning agent against multiple DevAI tasks
to get comprehensive performance metrics.

Usage:
    python scripts/run_planning_judge_batch.py \
        --conversation-files /path/to/conversations/*.jsonl \
        --devai-dir benchmark/devai/instances \
        --output-dir runs/batch_planning_evaluation \
        --filter-pattern "*Heart*" \
        --max-instances 10

Example:
    python scripts/run_planning_judge_batch.py \
        --conversation-files ../../artemis-agent/conversation_results_*.jsonl \
        --devai-dir benchmark/devai/instances \
        --output-dir runs/devai_batch_test \
        --max-instances 5
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import glob
from dotenv import load_dotenv

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_as_a_judge.planning_judge import PlanningJudgeAgent
from agent_as_a_judge.config import AgentConfig


def find_conversation_files(pattern: str) -> List[Path]:
    """Find conversation files matching the pattern."""
    files = [Path(f) for f in glob.glob(pattern)]
    return [f for f in files if f.exists()]


def find_devai_instances(devai_dir: Path, filter_pattern: str = "*") -> List[Path]:
    """Find DevAI instance files."""
    pattern = f"{filter_pattern}.json" if not filter_pattern.endswith('.json') else filter_pattern
    files = list(devai_dir.glob(pattern))
    return sorted(files)


def run_batch_evaluation(
    conversation_files: List[Path],
    devai_instances: List[Path],
    output_dir: Path,
    max_instances: int = None
) -> Dict[str, Any]:
    """Run batch evaluation."""

    results = {
        "total_evaluations": 0,
        "successful_evaluations": 0,
        "failed_evaluations": 0,
        "overall_satisfaction_rate": 0.0,
        "evaluation_details": [],
        "summary_by_category": {},
        "error_log": []
    }

    # Limit instances if specified
    if max_instances:
        devai_instances = devai_instances[:max_instances]

    total_evaluations = len(conversation_files) * len(devai_instances)
    print(f"📊 Planning to run {total_evaluations} evaluations")
    print(f"   Conversation files: {len(conversation_files)}")
    print(f"   DevAI instances: {len(devai_instances)}")

    config = AgentConfig()
    judge_dir = output_dir / "judge_workspace"
    judge_dir.mkdir(parents=True, exist_ok=True)

    evaluation_count = 0
    total_satisfied = 0
    total_requirements = 0

    for conv_file in conversation_files:
        for devai_instance in devai_instances:
            evaluation_count += 1

            try:
                print(f"\n[{evaluation_count}/{total_evaluations}] Evaluating:")
                print(f"  📄 {conv_file.name}")
                print(f"  🎯 {devai_instance.stem}")

                # Create planning judge
                planning_judge = PlanningJudgeAgent(
                    conversation_file=conv_file,
                    devai_instance=devai_instance,
                    judge_dir=judge_dir,
                    config=config
                )

                # Run evaluation
                eval_results = planning_judge.judge_planning_session()

                # Save individual results
                eval_name = f"{conv_file.stem}_{devai_instance.stem}"
                eval_output_file = output_dir / f"{eval_name}_judgment.json"
                planning_judge.save_judgment(eval_output_file)

                # Collect statistics
                satisfied = eval_results['satisfied_requirements']
                total = eval_results['total_requirements']

                total_satisfied += satisfied
                total_requirements += total
                results["successful_evaluations"] += 1

                # Store evaluation details
                results["evaluation_details"].append({
                    "conversation_file": conv_file.name,
                    "devai_instance": devai_instance.stem,
                    "satisfied_requirements": satisfied,
                    "total_requirements": total,
                    "satisfaction_rate": eval_results['satisfaction_rate'],
                    "evaluation_time": eval_results['total_time'],
                    "output_file": str(eval_output_file.relative_to(output_dir))
                })

                print(f"  ✅ {satisfied}/{total} requirements satisfied ({eval_results['satisfaction_rate']:.1%})")

            except Exception as e:
                error_msg = f"Failed to evaluate {conv_file.name} against {devai_instance.stem}: {e}"
                results["error_log"].append(error_msg)
                results["failed_evaluations"] += 1
                print(f"  ❌ Error: {e}")
                logging.error(error_msg)

    # Calculate overall statistics
    results["total_evaluations"] = evaluation_count
    if total_requirements > 0:
        results["overall_satisfaction_rate"] = total_satisfied / total_requirements

    # Save batch summary
    summary_file = output_dir / "batch_evaluation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=4)

    return results


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run Planning Judge Batch Evaluation")

    parser.add_argument(
        "--conversation-files",
        type=str,
        required=True,
        help="Glob pattern for conversation JSONL files (e.g., '../../artemis-agent/*.jsonl')"
    )

    parser.add_argument(
        "--devai-dir",
        type=str,
        required=True,
        help="Directory containing DevAI instance files"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for all judgment results"
    )

    parser.add_argument(
        "--filter-pattern",
        type=str,
        default="*",
        help="Filter DevAI instances by pattern (e.g., '*Heart*', '*ML*')"
    )

    parser.add_argument(
        "--max-instances",
        type=int,
        help="Maximum number of DevAI instances to evaluate (for testing)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level)

    # Find input files
    conversation_files = find_conversation_files(args.conversation_files)
    if not conversation_files:
        print(f"Error: No conversation files found matching: {args.conversation_files}")
        sys.exit(1)

    devai_dir = Path(args.devai_dir)
    if not devai_dir.exists():
        print(f"Error: DevAI directory not found: {devai_dir}")
        sys.exit(1)

    devai_instances = find_devai_instances(devai_dir, args.filter_pattern)
    if not devai_instances:
        print(f"Error: No DevAI instances found in {devai_dir} matching {args.filter_pattern}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Starting Batch Planning Judge Evaluation")
    print(f"   Found {len(conversation_files)} conversation files")
    print(f"   Found {len(devai_instances)} DevAI instances")
    print(f"   Output directory: {output_dir}")

    # Run batch evaluation
    try:
        results = run_batch_evaluation(
            conversation_files=conversation_files,
            devai_instances=devai_instances,
            output_dir=output_dir,
            max_instances=args.max_instances
        )

        # Print final summary
        print(f"\n🎉 Batch Evaluation Complete!")
        print(f"   Total Evaluations: {results['total_evaluations']}")
        print(f"   Successful: {results['successful_evaluations']}")
        print(f"   Failed: {results['failed_evaluations']}")
        print(f"   Overall Satisfaction Rate: {results['overall_satisfaction_rate']:.1%}")
        print(f"   Summary saved to: {output_dir / 'batch_evaluation_summary.json'}")

        if results['failed_evaluations'] > 0:
            print(f"\n⚠️  {results['failed_evaluations']} evaluations failed. Check error log in summary file.")

    except Exception as e:
        print(f"\n❌ Batch evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()