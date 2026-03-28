"""Evaluate confidence estimation against human judgments.

Metrics reported:
- Accuracy: correctness of AaaJ satisfied prediction vs human label.
- Mean confidence: average confidence assigned by AaaJ.
- AUROC: confidence ranking quality for correct vs incorrect predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark_dir",
        type=Path,
        required=True,
        help="Path to benchmark directory",
    )
    parser.add_argument(
        "--developer_agent",
        type=str,
        required=True,
        choices=["OpenHands", "MetaGPT", "GPT-Pilot"],
        help="Developer agent/framework name",
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="gray_box",
        help="AaaJ setting under agent_as_a_judge (e.g., gray_box)",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_confidence(aaaj_req: dict, aaaj_judge_stats: list | None, req_index: int) -> float:
    conf = aaaj_req.get("confidence")
    if conf is not None:
        return float(conf)

    sat_ratio = aaaj_req.get("satisfied_ratio")
    if sat_ratio is not None:
        sat_ratio = float(sat_ratio)
        return max(sat_ratio, 1.0 - sat_ratio)

    if aaaj_judge_stats and req_index < len(aaaj_judge_stats):
        stats = aaaj_judge_stats[req_index].get("llm_stats", {})
        if "confidence" in stats:
            return float(stats["confidence"])
        if "satisfied_ratio" in stats:
            sat_ratio = float(stats["satisfied_ratio"])
            return max(sat_ratio, 1.0 - sat_ratio)

    # Legacy fallback when only one vote is available.
    return 1.0


def build_pairs(aaaj: dict, human: dict) -> List[Tuple[int, float]]:
    aaaj_reqs = aaaj.get("requirements", [])
    human_reqs = human.get("requirements", [])
    judge_stats = aaaj.get("judge_stats", [])

    pairs: List[Tuple[int, float]] = []
    for i, (a, h) in enumerate(zip(aaaj_reqs, human_reqs)):
        a_label = a.get("satisfied")
        h_label = h.get("satisfied")

        if a_label is None or h_label is None:
            continue

        is_correct = 1 if bool(a_label) == bool(h_label) else 0
        confidence = infer_confidence(a, judge_stats, i)
        pairs.append((is_correct, confidence))

    return pairs


def accuracy(pairs: List[Tuple[int, float]]) -> float:
    return sum(label for label, _ in pairs) / len(pairs) if pairs else 0.0


def mean_confidence(pairs: List[Tuple[int, float]]) -> float:
    return sum(conf for _, conf in pairs) / len(pairs) if pairs else 0.0


def _average_rank(sorted_values: List[Tuple[float, int]], start: int, end: int) -> float:
    # Average rank for ties, ranks are 1-indexed.
    return (start + 1 + end) / 2.0


def auroc(pairs: List[Tuple[int, float]]) -> float:
    positives = sum(label for label, _ in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return 0.5

    scored = [(conf, label) for label, conf in pairs]
    scored.sort(key=lambda x: x[0])

    # Rank-based AUROC (Mann-Whitney U), tie-aware.
    rank_sum_pos = 0.0
    i = 0
    while i < len(scored):
        j = i
        while j + 1 < len(scored) and scored[j + 1][0] == scored[i][0]:
            j += 1
        avg_rank = _average_rank(scored, i, j)
        pos_in_group = sum(1 for k in range(i, j + 1) if scored[k][1] == 1)
        rank_sum_pos += avg_rank * pos_in_group
        i = j + 1

    return (rank_sum_pos - positives * (positives + 1) / 2.0) / (positives * negatives)


def main() -> None:
    args = parse_args()

    aaaj_dir = (
        args.benchmark_dir
        / "judgment"
        / args.developer_agent
        / "agent_as_a_judge"
        / args.setting
    )
    human_dir = (
        args.benchmark_dir
        / "judgment"
        / args.developer_agent
        / "human_as_a_judge"
    )

    aaaj_files = sorted(aaaj_dir.glob("*.json"))
    if not aaaj_files:
        raise FileNotFoundError(f"No files found in {aaaj_dir}")

    all_pairs: List[Tuple[int, float]] = []
    per_task = []

    for aaaj_file in aaaj_files:
        human_file = human_dir / aaaj_file.name
        if not human_file.exists():
            continue

        aaaj_data = load_json(aaaj_file)
        human_data = load_json(human_file)
        pairs = build_pairs(aaaj_data, human_data)
        if not pairs:
            continue

        task_acc = accuracy(pairs)
        task_conf = mean_confidence(pairs)
        task_auc = auroc(pairs)

        per_task.append((aaaj_file.stem, task_acc, task_conf, task_auc, len(pairs)))
        all_pairs.extend(pairs)

    if not all_pairs:
        raise RuntimeError("No comparable requirement labels found between AaaJ and human files")

    overall_acc = accuracy(all_pairs)
    overall_conf = mean_confidence(all_pairs)
    overall_auc = auroc(all_pairs)

    print("=" * 84)
    print(f"Confidence Evaluation - {args.developer_agent} ({args.setting})")
    print("=" * 84)
    print(f"Tasks evaluated           : {len(per_task)}")
    print(f"Requirements evaluated    : {len(all_pairs)}")
    print(f"Accuracy                  : {overall_acc:.4f}")
    print(f"Mean confidence           : {overall_conf:.4f}")
    print(f"AUROC                     : {overall_auc:.4f}")
    print("=" * 84)

    print("\nTop 10 lowest-AUROC tasks:")
    per_task.sort(key=lambda x: x[3])
    for name, task_acc, task_conf, task_auc, count in per_task[:10]:
        print(
            f"{name:<62} acc={task_acc:.3f} conf={task_conf:.3f} auc={task_auc:.3f} n={count}"
        )


if __name__ == "__main__":
    main()
