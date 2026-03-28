"""
DevAsk: A development tool for asking questions and checking requirements using LLM.
"""

import os
import logging
import warnings
from pathlib import Path
from dotenv import load_dotenv
from rich.logging import RichHandler
from agent_as_a_judge.llm.provider import LLM
from agent_as_a_judge.module.prompt.system_prompt_judge import get_judge_system_prompt
from agent_as_a_judge.module.prompt.prompt_judge import get_judge_prompt
from agent_as_a_judge.module.prompt.system_prompt_ask import get_ask_system_prompt
from agent_as_a_judge.module.prompt.prompt_ask import get_ask_prompt

warnings.simplefilter("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[RichHandler()],
)


class DevAsk:
    def __init__(self, workspace: Path, judge_dir: Path):
        self.workspace = workspace
        self.judge_dir = judge_dir
        self.llm = self._initialize_llm()

    def _initialize_llm(self) -> LLM:
        try:
            model = os.getenv("DEFAULT_LLM")
            api_key = os.getenv("OPENAI_API_KEY")
            return LLM(model=model, api_key=api_key)
        except KeyError as e:
            logging.error(f"Missing environment variable: {e}")
            raise

    def check(
        self,
        criteria: str,
        evidence: str,
        majority_vote: int = 1,
        critical_threshold: float = 0.5,
    ) -> dict:
        total_llm_stats = self._initialize_llm_stats()
        responses, judges = self._collect_judgments(
            criteria, evidence, majority_vote, total_llm_stats
        )

        satisfied_count = judges.count("<SATISFIED>")
        total_judges = len(judges)
        satisfied_ratio = (satisfied_count / total_judges) if total_judges else 0.0
        majority_judge = (satisfied_count / total_judges) >= critical_threshold
        confidence = max(satisfied_ratio, 1.0 - satisfied_ratio)

        total_llm_stats.update(
            {
                "satisfied": majority_judge,
                "reason": responses,
                "confidence": round(confidence, 4),
                "satisfied_ratio": round(satisfied_ratio, 4),
                "majority_vote": total_judges,
                "critical_threshold": critical_threshold,
            }
        )
        return total_llm_stats

    def _collect_judgments(
        self, criteria: str, evidence: str, majority_vote: int, llm_stats: dict
    ) -> tuple:
        system_prompt = get_judge_system_prompt(language="English")
        prompt = get_judge_prompt(criteria=criteria, evidence=evidence)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        responses = []
        judges = []

        for _ in range(majority_vote):
            result = self.llm._llm_inference(messages)
            responses.append(result["llm_response"])
            judges.append(self._parse_judge(result["llm_response"]))

            self._update_llm_stats(llm_stats, result)

        return responses, judges

    @staticmethod
    def _parse_judge(response: str) -> str:
        if "<SATISFIED>" in response:
            return "<SATISFIED>"
        return "<UNSATISFIED>"

    @staticmethod
    def _initialize_llm_stats() -> dict:
        return {
            "cost": 0.0,
            "inference_time": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    @staticmethod
    def _update_llm_stats(stats: dict, result: dict):
        stats["cost"] += result.get("cost", 0)
        stats["inference_time"] += result.get("inference_time", 0)
        stats["input_tokens"] += result.get("input_tokens", 0)
        stats["output_tokens"] += result.get("output_tokens", 0)

    def ask(self, question: str, evidence: str) -> str:
        if not evidence:
            raise ValueError("Evidence must be provided.")

        system_prompt = get_ask_system_prompt(language="English")
        prompt = get_ask_prompt(evidence=evidence, question=question)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        result = self.llm._llm_inference(messages)
        return result["llm_response"]


if __name__ == "__main__":
    load_dotenv()
    workspace_path = (
        Path(os.getenv("PROJECT_DIR"))
        / "benchmark/workspace/OpenHands/39_Drug_Response_Prediction_SVM_GDSC_ML"
    )
    judge_dir = (
        Path(os.getenv("PROJECT_DIR"))
        + "/benchmark/judgement/OpenHands/39_Drug_Response_Prediction_SVM_GDSC_ML"
    )
    judge_dir.mkdir(parents=True, exist_ok=True)
    dev_ask = DevAsk(workspace=workspace_path, judge_dir=judge_dir)

    while True:
        user_query = input("Ask your question: ")
        if user_query.lower() in ["exit", "quit"]:
            break
        try:
            response = dev_ask.ask(question="This is a test.", evidence=user_query)
            print(response)
        except Exception as e:
            logging.error(f"An error occurred: {e}")
