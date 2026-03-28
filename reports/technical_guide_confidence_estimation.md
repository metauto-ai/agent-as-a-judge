# Technical Guide: Confidence Estimation in Agent-as-a-Judge

## 1. Goal and Scope

This guide explains the confidence-estimation extension added to the Agent-as-a-Judge (AaaJ) pipeline, with enough detail for readers comfortable with Python pipelines and evaluation metrics.

Objective:
- produce requirement-level confidence for binary judgments
- measure whether confidence is informative using Accuracy, Mean Confidence, and AUROC

Evaluation target:
- compare AaaJ binary outputs against human labels per requirement
- treat agreement with human as correctness

## 2. Pipeline Design

### 2.1 Requirement-Level Voting

For each requirement, the judge runs multiple independent LLM checks.

Parameters:
- k = majority_vote
- tau = critical_threshold

Let s be the number of SATISFIED votes out of k.

Definitions:
- satisfied_ratio = s / k
- y_hat = 1[satisfied_ratio >= tau]
- confidence = max(satisfied_ratio, 1 - satisfied_ratio)

Notes:
- confidence is symmetric around 0.5
- high agreement (close to 0 or 1 ratio) gives high confidence
- split votes (near 0.5 ratio) give low confidence

### 2.2 Stored Fields in Outputs

Each requirement now carries:
- satisfied
- confidence
- satisfied_ratio
- majority_vote
- critical_threshold

This schema supports post-hoc metric computation without re-running inference.

## 3. Code Integration Map

Implemented components:
- [agent_as_a_judge/config.py](agent_as_a_judge/config.py): adds majority_vote and critical_threshold in AgentConfig
- [agent_as_a_judge/module/ask.py](agent_as_a_judge/module/ask.py): computes vote ratio, final label, and confidence inside check()
- [agent_as_a_judge/agent.py](agent_as_a_judge/agent.py): propagates vote params and persists confidence metadata into requirement records
- [scripts/run_aaaj.py](scripts/run_aaaj.py): exposes --majority_vote and --critical_threshold as CLI flags
- [scripts/evaluate_confidence.py](scripts/evaluate_confidence.py): computes evaluation metrics against human judgments

## 4. Metric Definitions

### 4.1 Correctness Label

For each requirement i:
- c_i = 1 if AaaJ label equals human label
- c_i = 0 otherwise

### 4.2 Accuracy

Accuracy = (1/N) * sum_i c_i

Interpretation:
- standard agreement with human labels
- does not use confidence magnitude

### 4.3 Mean Confidence

MeanConfidence = (1/N) * sum_i conf_i

Interpretation:
- average claimed certainty
- useful when compared to Accuracy

Calibration tendency check:
- MeanConfidence > Accuracy suggests overconfidence
- MeanConfidence < Accuracy suggests underconfidence

### 4.4 AUROC

Inputs to AUROC:
- binary class: c_i in {0,1}
- score: conf_i

Interpretation:
- probability that a random correct case has higher confidence than a random incorrect case

Range:
- 1.0: perfect ranking
- 0.5: random ranking
- < 0.5: inverted ranking tendency

Implementation note:
- [scripts/evaluate_confidence.py](scripts/evaluate_confidence.py) computes AUROC using a rank-based Mann-Whitney U formulation with tie handling, without external ML libraries.

## 5. Available Experimental Outputs

Current aggregate outputs on available files:

| Framework | Accuracy | Mean Confidence | AUROC |
|---|---:|---:|---:|
| OpenHands | 0.9016 | 1.0000 | 0.4861 |
| MetaGPT | 0.9208 | 1.0000 | 0.4828 |
| GPT-Pilot | 0.8661 | 1.0000 | 0.4898 |

## 6. Why AUROC is Near 0.5 in Current Results

Observed behavior is expected for legacy outputs.

Root cause:
- prior result files effectively contain constant confidence (or fallback confidence)
- constant scores cannot rank positives above negatives
- AUROC then collapses toward random (about 0.5)

Therefore:
- this is not primarily a metric bug
- it is a data-generation issue (insufficient confidence variance)

## 7. Recommended Experiment Sequence

### Experiment 1 (already done): Legacy baseline
- evaluate current files
- establish baseline metrics and identify confidence collapse

### Experiment 2 (next): Multi-vote rerun
- regenerate judgments with majority_vote >= 5
- use a new setting name to avoid overwriting baseline files

Expected effect:
- confidence distribution spreads (for example 0.6, 0.8, 1.0)
- AUROC becomes informative

### Experiment 3 (optional but strong): Calibration diagnostics
- add ECE and reliability bins
- report calibration curve and confidence-accuracy gap by bin

## 8. Minimal Reproducibility Checklist

1. Run AaaJ with explicit vote configuration.
2. Confirm output requirements include confidence-related fields.
3. Run [scripts/evaluate_confidence.py](scripts/evaluate_confidence.py) for each framework and setting.
4. Report Accuracy, Mean Confidence, AUROC side by side.
5. Interpret AUROC only after confirming non-trivial confidence variance.

## 9. Common Failure Modes

- Constant confidence across all items:
  AUROC becomes non-informative even if Accuracy is high.

- Misreading high Accuracy as good calibration:
  Accuracy and calibration are different properties.

- Overwriting baseline setting outputs:
  loses comparison value; always keep separate setting names.

## 10. Practical Conclusion

The implementation is integrated and evaluation-ready. The main blocker for meaningful AUROC claims is not code correctness but dataset regeneration with multi-vote inference so confidence scores carry ranking information.
