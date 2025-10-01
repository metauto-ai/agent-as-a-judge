#!/usr/bin/env python3
"""
Script to run the Planning Judge Agent on Artemis planning outputs.

Usage:
    python scripts/run_planning_judge.py \
        --conversation-file /path/to/conversation_results.jsonl \
        --devai-instance /path/to/devai_instance.json \
        --output-dir /path/to/output \
        --judge-name "my_planning_evaluation"

Example:
    python scripts/run_planning_judge.py \
        --conversation-file ../../artemis-agent/conversation_results_1c18c90a-5a02-4755-bfc3-8b92d8147860.jsonl \
        --devai-instance benchmark/devai/instances/17_Heart_Disease_Prediction_XGBoost_UCI_ML.json \
        --output-dir runs/planning_judge_test \
        --judge-name "heart_disease_planning"
"""

import argparse
import logging
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        pass

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_as_a_judge.planning_judge import PlanningJudgeAgent
from agent_as_a_judge.config import AgentConfig

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run Planning Judge Agent")

    parser.add_argument(
        "--conversation-file",
        type=str,
        required=True,
        help="Path to Artemis planning conversation JSONL file"
    )

    parser.add_argument(
        "--devai-instance",
        type=str,
        required=True,
        help="Path to DevAI instance JSON file"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for judgment results"
    )

    parser.add_argument(
        "--judge-name",
        type=str,
        required=True,
        help="Name for this judgment run"
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

    # Validate input files
    conversation_file = Path(args.conversation_file)
    if not conversation_file.exists():
        print(f"Error: Conversation file not found: {conversation_file}")
        sys.exit(1)

    devai_instance = Path(args.devai_instance)
    if not devai_instance.exists():
        print(f"Error: DevAI instance not found: {devai_instance}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    judge_dir = output_dir / "judge_workspace"
    judge_dir.mkdir(parents=True, exist_ok=True)

    # Initialize planning judge
    config = AgentConfig()

    planning_judge = PlanningJudgeAgent(
        conversation_file=conversation_file,
        devai_instance=devai_instance,
        judge_dir=judge_dir,
        config=config
    )

    # Run judgment
    try:
        print(f"\n🔍 Starting Planning Judge Evaluation")
        print(f"   Conversation: {conversation_file.name}")
        print(f"   DevAI Task: {devai_instance.stem}")
        print(f"   Output: {output_dir}")

        # Run the judgment
        results = planning_judge.judge_planning_session()

        # Save results
        output_file = output_dir / f"{args.judge_name}_judgment.json"
        planning_judge.save_judgment(output_file)

        # Print summary
        print(f"\n✅ Planning Judge Evaluation Complete!")
        print(f"   Requirements Satisfied: {results['satisfied_requirements']}/{results['total_requirements']}")
        print(f"   Satisfaction Rate: {results['satisfaction_rate']:.2%}")
        print(f"   Total Time: {results['total_time']:.2f}s")
        print(f"   Results saved to: {output_file}")

        # Print detailed results if verbose
        if args.verbose:
            print(f"\n📊 Detailed Results:")
            for stat in results['judge_stats']:
                status = "✅" if stat['satisfied'] else "❌"
                print(f"   {status} Req {stat['requirement_id']}: {stat['category']}")
                if not stat['satisfied']:
                    print(f"      Reason: {stat['reason'][:100]}...")

    except Exception as e:
        print(f"\n❌ Error running Planning Judge: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()