# Beginner Guide: Confidence Estimation in Agent-as-a-Judge

## 1) What problem are we solving?

Your system (Agent-as-a-Judge, or AaaJ) checks whether each requirement is satisfied.

Before our changes, it mostly gave a binary answer:
- satisfied = true
- satisfied = false

That is useful, but incomplete.

As a researcher, you also want to know:
- How sure is the system about each decision?
- Can we trust high-confidence decisions more than low-confidence ones?

This is what confidence estimation and calibration are about.

## 2) Core ideas (very simple)

### Prediction
A binary decision for each requirement:
- true (satisfied)
- false (not satisfied)

### Confidence score
A number between 0 and 1 that says how certain the system is.
- close to 1.0 = very sure
- close to 0.5 = uncertain

### Correctness label
To evaluate quality, compare AaaJ with human judgment:
- correct = 1 if AaaJ matches human
- incorrect = 0 otherwise

## 3) What we implemented in this repo

We added majority-vote confidence.

### Step A: Ask the judge multiple times
Instead of one LLM judgment, ask k times (for example, k = 5).

### Step B: Compute vote ratio
If 4 out of 5 votes say satisfied:
- satisfied_ratio = 4 / 5 = 0.8

### Step C: Final binary decision
Use threshold tau (usually 0.5):
- satisfied = (satisfied_ratio >= tau)

### Step D: Confidence
We use:

confidence = max(satisfied_ratio, 1 - satisfied_ratio)

Examples:
- votes 5/5 satisfied -> confidence = 1.0
- votes 4/5 satisfied -> confidence = 0.8
- votes 3/5 satisfied -> confidence = 0.6
- votes 2/5 satisfied -> confidence = 0.6 (now leaning not satisfied)

Interpretation:
- agreement among votes gives higher confidence
- split votes give lower confidence

## 4) Where this was added in code

Main implementation locations:
- [agent_as_a_judge/config.py](agent_as_a_judge/config.py): added majority_vote and critical_threshold
- [agent_as_a_judge/module/ask.py](agent_as_a_judge/module/ask.py): computes satisfied_ratio and confidence
- [agent_as_a_judge/agent.py](agent_as_a_judge/agent.py): saves confidence fields in each requirement result
- [scripts/run_aaaj.py](scripts/run_aaaj.py): exposes CLI arguments for majority voting
- [scripts/evaluate_confidence.py](scripts/evaluate_confidence.py): evaluates Accuracy, Mean Confidence, AUROC

## 5) Metrics you are seeing

### Accuracy
How often AaaJ matches human judgment.

Accuracy = (# correct decisions) / (# total decisions)

### Mean Confidence
Average confidence value across all requirements.

This tells you how "sure" the system claims to be overall.

### AUROC (most confusing one)
AUROC checks ranking quality of confidence.

Question AUROC asks:
- "Do correct predictions usually get higher confidence than incorrect predictions?"

Equivalent interpretation:
- AUROC is the probability that a random correct case has higher confidence than a random incorrect case.

Meaning:
- 1.0 = perfect ranking
- 0.5 = random / no useful confidence signal
- < 0.5 = confidence is often backwards

## 6) Why your current AUROC is around 0.48 to 0.49

This is expected for your current files.

Reason:
- older outputs effectively had constant confidence (often fallback to 1.0)
- when almost all scores are the same, confidence cannot rank correct vs incorrect
- AUROC then becomes near random (~0.5)

So this does NOT mean your code is broken.
It means the dataset being evaluated does not yet contain informative confidence variation.

## 7) What to run next (important)

To get meaningful confidence behavior, regenerate judgments with multi-vote enabled.

Example target:
- majority_vote = 5
- critical_threshold = 0.5
- new setting name like gray_box_conf5

Then run [scripts/evaluate_confidence.py](scripts/evaluate_confidence.py) on that new setting.

Expected outcome:
- confidence values will vary (0.6, 0.8, 1.0, ...)
- AUROC should become informative (ideally > 0.5)

## 8) Practical reading guide for your results

When you see results, interpret them in this order:

1. Accuracy: Is basic judging quality acceptable?
2. AUROC: Is confidence ranking useful?
3. Mean Confidence vs Accuracy gap:
- if Mean Confidence >> Accuracy -> overconfident
- if Mean Confidence << Accuracy -> underconfident
- if close -> better calibration tendency

## 9) Current limitations

- Current reported AUROC came from legacy outputs without real confidence variation.
- No ECE/reliability diagram yet.
- Majority-vote confidence is simple and practical, but not the only method.

## 10) One-sentence summary

You successfully added confidence estimation into AaaJ; now the key step is to generate new multi-vote outputs so AUROC can reflect real confidence quality instead of legacy constant-score behavior.
