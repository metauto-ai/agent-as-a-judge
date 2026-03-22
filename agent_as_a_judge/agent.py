import io
import os
import time
import json
import pickle
import logging
from pathlib import Path
from dataclasses import dataclass
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.text import Text
from rich.markdown import Markdown
from rich.panel import Panel
from rich.emoji import Emoji

from agent_as_a_judge.module.code_search import DevCodeSearch
from agent_as_a_judge.module.read import DevRead
from agent_as_a_judge.module.graph import DevGraph
from agent_as_a_judge.module.ask import DevAsk
from agent_as_a_judge.module.locate import DevLocate
from agent_as_a_judge.module.text_retrieve import DevTextRetrieve
from agent_as_a_judge.module.memory import Memory
from agent_as_a_judge.module.planning import Planning
from agent_as_a_judge.llm.provider import LLM
from agent_as_a_judge.config import AgentConfig
from agent_as_a_judge.utils import truncate_string


console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[RichHandler()],
)


class JudgeAgent:
    """
    This proof-of-concept shows how Agent-as-a-Judge can evaluate the workspace
    and behaviour of other developer agents.
    """

    def __init__(
        self,
        workspace: Path,
        instance: Path,
        judge_dir: Path,
        config: AgentConfig,
        trajectory_file: Path = None,
    ):

        self.workspace = workspace
        self.instance = instance
        self.judge_dir = judge_dir
        self.trajectory_file = trajectory_file
        self.config = config

        self.llm = LLM(
            model=os.getenv("DEFAULT_LLM"), api_key=os.getenv("OPENAI_API_KEY")
        )

        # Paths for Judge-specific directories and files
        self.judge_workspace = Path(
            judge_dir, os.path.basename(os.path.normpath(self.workspace))
        )
        self.judge_workspace.mkdir(parents=True, exist_ok=True)
        self.graph_file = self.judge_workspace / "graph.pkl"
        self.tags_file = self.judge_workspace / "tags.json"
        self.structure_file = self.judge_workspace / "tree_structure.json"

        if (
            not self.graph_file.exists()
            or not self.tags_file.exists()
            or not self.structure_file.exists()
        ):
            self.construct_graph()

        self.structure = self.aaaj_search.load_structure()
        self.judge_stats = []
        self.total_time = 0.0

        self._initialize_class_vars()
        # instance_data = self._load_instance_data()
        # self.aaaj_memory = Memory(self.judge_dir / f"{instance_data['name']}.json")

        # Construct the codebase graph if not already saved
        if (
            not self.graph_file.exists()
            or not self.tags_file.exists()
            or not self.structure_file.exists()
        ):
            self.construct_graph()

    @property
    def aaaj_graph(self):
        if not hasattr(self, "_aaaj_graph"):
            self._aaaj_graph = DevGraph(
                root=str(self.workspace),
                include_dirs=self.config.include_dirs,
                exclude_dirs=self.config.exclude_dirs,
                exclude_files=self.config.exclude_files,
            )
        return self._aaaj_graph

    @property
    def aaaj_search(self):
        if not hasattr(self, "_aaaj_search"):
            self._aaaj_search = DevCodeSearch(
                str(self.judge_workspace), self.config.setting
            )
        return self._aaaj_search

    @property
    def aaaj_read(self):
        if not hasattr(self, "_aaaj_read"):
            self._aaaj_read = DevRead()
        return self._aaaj_read

    @property
    def aaaj_ask(self):
        if not hasattr(self, "_aaaj_ask"):
            self._aaaj_ask = DevAsk(self.workspace, self.judge_workspace)
        return self._aaaj_ask

    @property
    def aaaj_locate(self):
        if not hasattr(self, "_aaaj_locate"):
            self._aaaj_locate = DevLocate()
        return self._aaaj_locate

    @property
    def aaaj_memory(self):
        instance_data = self._load_instance_data()
        if instance_data and "name" in instance_data:
            if not hasattr(self, "_aaaj_memory"):
                self._aaaj_memory = Memory(
                    self.judge_dir / f"{instance_data['name']}.json"
                )
            return self._aaaj_memory
        else:
            return None

    @property
    def aaaj_retrieve(self):
        if not hasattr(self, "_aaaj_retrieve"):
            self._aaaj_retrieve = DevTextRetrieve(str(self.trajectory_file))
        return self._aaaj_retrieve

    @staticmethod
    def _initialize_class_vars():

        if not hasattr(JudgeAgent, "total_check"):
            JudgeAgent.total_check = 0

    def judge_anything(self):

        logging.info(f"Judging requirements for instance: {self.instance.name}")
        instance_data = self._load_instance_data()
        total_checked_requirements = 0

        for i, requirement in enumerate(instance_data.get("requirements", [])):
            user_query = instance_data.get("query", "")
            criteria = requirement["criteria"]

            if self.config.planning == "planning":
                self.planning = Planning()
                planning_result = self.planning.generate_plan(criteria)
                workflow = planning_result["actions"]
                planning_llm_stats = planning_result["llm_stats"]

            elif self.config.planning == "comprehensive (no planning)":
                workflow = [
                    "user_query",
                    "workspace",
                    "locate",
                    "read",
                    "search",
                    "history",
                    "trajectory",
                ]
                planning_llm_stats = None

            elif self.config.planning == "efficient (no planning)":
                workflow = ["workspace", "locate", "read", "trajectory"]
                planning_llm_stats = None

            if self.config.setting == "black_box" and "trajectory" in workflow:
                workflow.remove("trajectory")

            llm_stats, total_time = self.check_requirement(
                criteria, workflow, user_query=user_query
            )

            # Persist requirement-level prediction metadata for downstream confidence analysis.
            requirement["satisfied"] = llm_stats["satisfied"]
            requirement["confidence"] = llm_stats.get("confidence")
            requirement["satisfied_ratio"] = llm_stats.get("satisfied_ratio")
            requirement["majority_vote"] = llm_stats.get("majority_vote")
            requirement["critical_threshold"] = llm_stats.get("critical_threshold")

            if planning_llm_stats:
                llm_stats["input_tokens"] += planning_llm_stats.get("input_tokens", 0)
                llm_stats["output_tokens"] += planning_llm_stats.get("output_tokens", 0)
                llm_stats["cost"] += planning_llm_stats.get("cost", 0)
                llm_stats["inference_time"] += planning_llm_stats.get(
                    "inference_time", 0
                )

            judgment_entry = {
                "requirement_index": i,
                "criteria": criteria,
                "satisfied": llm_stats["satisfied"],
                "llm_stats": llm_stats,
                "total_time": total_time,
            }
            self.judge_stats.append(judgment_entry)
            total_checked_requirements += 1
            JudgeAgent.total_check += 1
            self._save_judgment_data(instance_data)

        logging.info(f"Total requirements checked: {total_checked_requirements}")

    def ask_anything(self, question: str):

        workflow = ["workspace", "locate", "read", "search"]
        llm_stats, start_time = self.check_requirement(
            criteria=question, workflow=workflow, user_query=question
        )
        evidence = truncate_string(
            self.display_tree(), model=self.llm.model_name, max_tokens=10000
        )
        answer = self.aaaj_ask.ask(question, evidence=evidence)
        total_time = time.time() - start_time
        return answer

    def check_requirement(self, criteria: str, workflow: list, user_query: str):

        start_time = time.time()
        total_llm_stats = {
            "cost": 0.0,
            "inference_time": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        combined_evidence = ""
        related_files = []

        workspace_info = truncate_string(
            self.display_tree(), model=self.llm.model_name, max_tokens=2000
        )

        for info_type in workflow:
            if info_type == "user_query" and user_query:
                combined_evidence += (
                    f">>> [Reference] Original User Query:\n\n{user_query}\n\n"
                )
                logging.info(
                    f">>> [Reference] Original User Query:\n\n{user_query}\n\n"
                )

            elif info_type == "workspace":
                combined_evidence += (
                    f">>> [Key Evidence] Workspace Structure:\n\n{workspace_info}\n\n"
                )
                # logging.info(f">>> [Key Evidence] Workspace Structure:\n\n{workspace_info}\n\n")

            elif info_type == "locate":
                locate_result = self.locate_file(criteria, workspace_info)
                related_files = locate_result["file_paths"]
                total_llm_stats.update(locate_result["llm_stats"])
                logging.info(
                    f">>> [Reference] Located Files:\n\n{locate_result['file_paths']}\n\n"
                )

            elif info_type == "read" and related_files:
                for file_path in related_files:
                    content, llm_stats = self.aaaj_read.read(Path(file_path))
                    combined_evidence += f">>> [Key Evidence] Content of Files:\n\nContent of {file_path}:\n```\n{truncate_string(content, model=self.llm.model_name, max_tokens=2000)}\n```\n"
                    logging.info(
                        f">>> [Key Evidence] Content of Files:\n\nContent of {file_path}:\n```\n{truncate_string(content, model=self.llm.model_name, max_tokens=2000)}\n```\n"
                    )
                    if llm_stats:
                        total_llm_stats.update(llm_stats)

            elif info_type == "search":
                search_list = self.aaaj_search.search(criteria, search_type="embedding")
                for search_context in search_list:
                    combined_evidence += f">>> [Reference] Relevant Search Evidence:\n\n{self.aaaj_search.display(search_context)}\n\n"
                    logging.info(
                        f">>> [Reference] Relevant Search Evidence:\n\n{self.aaaj_search.display(search_context)}\n\n"
                    )

            elif info_type == "history":
                if self.aaaj_memory:
                    historical_evidence = self.aaaj_memory.get_historical_evidence()
                    combined_evidence += f">>> [Reference] Historical Judgments:\n\n{historical_evidence}\n\n"
                    logging.info(
                        f">>> [Reference] Historical Judgments:\n\n{historical_evidence}\n\n"
                    )
                else:
                    logging.warning(
                        ">>> [Reference] No historical evidence available (aaaj_memory is None)"
                    )

            elif info_type == "trajectory":
                llm_trajectory_stats = self.aaaj_retrieve.llm_summary(criteria)
                combined_evidence += f">>> [Reference] Trajectory Evidence:\n\n{llm_trajectory_stats.get('trajectory_analysis', '')}\n\n"
                logging.info(
                    f">>> [Reference] Trajectory Evidence:\n\n{llm_trajectory_stats.get('trajectory_analysis', '')}\n\n"
                )
                total_llm_stats.update(llm_trajectory_stats)

        combined_evidence = truncate_string(
            combined_evidence, model=self.llm.model_name
        )
        check_llm_stats = self.aaaj_ask.check(
            criteria,
            combined_evidence,
            majority_vote=self.config.majority_vote,
            critical_threshold=self.config.critical_threshold,
        )
        total_llm_stats.update(check_llm_stats)
        total_time = time.time() - start_time

        self.display_judgment(
            criteria=criteria,
            satisfied=check_llm_stats["satisfied"],
            reason=check_llm_stats["reason"],
            logger=logging,
        )

        return total_llm_stats, total_time

    def construct_graph(self):

        filepaths = self.aaaj_graph.list_py_files([str(self.workspace)])
        tags, graph = self.aaaj_graph.build(filepaths) if filepaths else (None, None)
        self._save_graph_and_tags(graph, tags)
        self._save_file_structure()

    def display_tree(self, max_depth: int = None) -> str:

        def add_branch(tree: Tree, structure: dict, current_depth: int):
            if max_depth is not None and current_depth > max_depth:
                return
            for key, value in structure.items():
                if isinstance(value, dict):
                    branch = tree.add(f"{key}")
                    add_branch(branch, value, current_depth + 1)
                else:
                    file_label = (
                        f"[bold white]{key}[/bold white]"
                        if value
                        else f"[dim]{key}[/dim]"
                    )
                    tree.add(file_label)

        tree = Tree("[bold blue]Project Structure[/bold blue]")
        add_branch(tree, self.structure["tree_structure"], current_depth=0)

        metadata = Text.from_markup(
            f"[bold cyan]Workspace Path:[/bold cyan] [bold white]{self.workspace}[/bold white]\n"
            f"[bold cyan]Total Nodes:[/bold cyan] [bold white]{len(self.structure['tree_structure'])}[/bold white]\n"
        )

        table = Table.grid(expand=True)
        table.add_column(justify="left")
        table.add_row(metadata)
        table.add_row(tree)

        combined_panel = Panel(
            table,
            title="[bold magenta]Project Tree[/bold magenta]",
            border_style="bold blue",
            title_align="left",
            padding=(1, 2),
            expand=True,
        )

        with io.StringIO() as buf:
            console = Console(record=True, width=120)
            console.print(combined_panel, soft_wrap=True)
            return console.export_text()

    def _save_graph_and_tags(self, graph, tags):

        logging.info("Saving the graph and tags...")
        with open(self.graph_file, "wb") as f:
            pickle.dump(graph, f)
        with open(self.tags_file, "w") as f:
            json.dump(
                (
                    [
                        {
                            "fname": tag.fname,
                            "rel_fname": tag.rel_fname,
                            "line_number": tag.line,
                            "name": tag.name,
                            "identifier": tag.identifier,
                            "category": tag.category,
                            "details": tag.details,
                        }
                        for tag in tags
                    ]
                    if tags
                    else {}
                ),
                f,
                indent=4,
            )

    def _save_file_structure(self):

        def build_tree_structure(current_path):
            tree = {}
            for root, dirs, files in os.walk(current_path):
                dirs[:] = [
                    d
                    for d in dirs
                    if not any(excluded in d for excluded in self.config.exclude_dirs)
                ]
                relative_root = os.path.relpath(root, self.workspace)
                tree[relative_root] = {
                    file: None
                    for file in files
                    if not file.startswith(".")
                    and file not in self.config.exclude_files
                }
            return tree

        tree_structure = build_tree_structure(self.workspace)
        self.workspace_info = {
            "workspace": str(self.workspace),
            "tree_structure": tree_structure,
        }
        save_path = self.judge_workspace / "tree_structure.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.workspace_info, f, indent=4)

    def locate_file(self, criteria: str, workspace_info: str) -> dict:

        return self.aaaj_locate.locate_file(criteria, workspace_info)

    def display_judgment(
        self, criteria: str, satisfied: bool, reason: str, logger: logging.Logger
    ):

        criteria_markdown = f"{Emoji('question')} **Criteria**\n{criteria}"
        satisfied_markdown = f"{Emoji('white_check_mark' if satisfied else 'x')} **Satisfied**: {satisfied}"
        reason_markdown = f"{Emoji('thought_balloon')} **Reason**\n{reason}"

        panel_content = f"{criteria_markdown}\n\n---\n\n{satisfied_markdown}\n\n---\n\n{reason_markdown}"

        panel = Panel(
            Markdown(panel_content),
            title="[bold magenta]📝 Judgment Result[/bold magenta]",
            border_style="bold cyan",
            title_align="center",
            padding=(1, 2),
        )

        with io.StringIO() as buf:
            temp_console = Console(file=buf, width=80, record=True)
            temp_console.print(panel)
            formatted_message = buf.getvalue()

        console.print(panel)
        logger.info(f"Judgment Details:\n{formatted_message}")

    def _save_judgment_data(self, instance_data):

        for judgment in self.judge_stats:
            if "llm_stats" in judgment and "llm_response" in judgment["llm_stats"]:
                del judgment["llm_stats"]["llm_response"]
            if (
                "llm_stats" in judgment
                and "trajectory_analysis" in judgment["llm_stats"]
            ):
                del judgment["llm_stats"]["trajectory_analysis"]

        output_file = self.judge_dir / self.instance.name
        instance_data["judge_stats"] = self.judge_stats
        with open(output_file, "w") as f:
            json.dump(instance_data, f, indent=4)

    def _load_instance_data(self):

        if self.instance:
            with open(self.instance, "r") as f:
                return json.load(f)
