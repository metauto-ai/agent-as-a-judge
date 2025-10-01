"""
PlanningJudgeAgent: Evaluates planning agent outputs against DevAI instances.

Instead of evaluating implemented code, this judges whether:
1. Plans include necessary tasks
2. Dependencies are logical
3. Generated prompts are comprehensive
4. Task decomposition covers all requirements
"""

import json
import logging
import time
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from rich.logging import RichHandler
from rich.console import Console
import networkx as nx

from agent_as_a_judge.llm.provider import LLM
from agent_as_a_judge.module.ask import DevAsk
from agent_as_a_judge.module.memory import Memory
from agent_as_a_judge.module.planning import Planning
from agent_as_a_judge.config import AgentConfig
from agent_as_a_judge.utils import truncate_string

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[RichHandler()],
)


class ConversationParser:
    """Parse Artemis planning agent conversation JSONL files."""

    def __init__(self, conversation_file: Path):
        self.conversation_file = conversation_file

    def extract_planning_data(self) -> Dict[str, Any]:
        """Extract plan structure, order, prompts from conversation file."""
        try:
            with open(self.conversation_file, 'r') as f:
                content = f.read().strip()

            # Try to parse as single JSON object first
            try:
                final_state = json.loads(content)
            except json.JSONDecodeError:
                # If that fails, try JSONL format (read last line)
                lines = content.split('\n')
                # Find the last non-empty line that contains JSON
                for line in reversed(lines):
                    line = line.strip()
                    if line and line.startswith('{'):
                        final_state = json.loads(line)
                        break
                else:
                    raise ValueError("No valid JSON found in file")

            # Extract planning outputs from artemis_state.ui_models.plan.response
            ui_models = final_state.get("artemis_state", {}).get("ui_models", {})
            plan_response = ui_models.get("plan", {}).get("response", {})

            return {
                "plan": plan_response.get("plan", {}),
                "order": plan_response.get("order", []),
                "prompts": plan_response.get("prompts", {}),
                "session_complete": final_state.get("session_complete", False),
                "original_task": final_state.get("original_user_task", "")
            }
        except Exception as e:
            logging.error(f"Failed to parse conversation file {self.conversation_file}: {e}")
            return {}


class PlanAnalyzer:
    """Analyze planning agent outputs for structure and dependencies."""

    def analyze_plan_structure(self, plan_data: Dict) -> str:
        """Analyze the hierarchical plan structure."""
        if not plan_data or "root" not in plan_data:
            return "No plan structure found"

        def analyze_node(node: Dict, depth: int = 0) -> List[str]:
            indent = "  " * depth
            analysis = [f"{indent}- {node.get('name', 'Unnamed')} (ID: {node.get('id')}, Level: {node.get('level')})"]

            # Add description if available
            if desc := node.get('description'):
                analysis.append(f"{indent}  Desc: {desc[:100]}...")

            # Analyze children recursively
            for child in node.get('children', []):
                analysis.extend(analyze_node(child, depth + 1))

            return analysis

        analysis_lines = ["Plan Structure Analysis:"]
        for root_node in plan_data.get("root", []):
            analysis_lines.extend(analyze_node(root_node))

        return "\n".join(analysis_lines)

    def analyze_dependencies(self, plan_data: Dict, order: List[int]) -> str:
        """Analyze task dependencies and ordering."""
        if not plan_data or not order:
            return "No dependency information available"

        # Build task lookup
        task_lookup = {}
        def build_lookup(node: Dict):
            task_lookup[node.get('id')] = node
            for child in node.get('children', []):
                build_lookup(child)

        for root_node in plan_data.get("root", []):
            build_lookup(root_node)

        # Analyze ordering
        analysis = ["Dependency Analysis:"]
        analysis.append(f"Task execution order: {order}")

        for i, task_id in enumerate(order):
            if task_id in task_lookup:
                task = task_lookup[task_id]
                depends_on = task.get('depends_on', set())
                if isinstance(depends_on, str) and depends_on.startswith('set('):
                    # Parse set string representation
                    depends_on = set()

                analysis.append(f"{i+1}. {task.get('name')} (ID: {task_id})")
                if depends_on:
                    analysis.append(f"   Dependencies: {depends_on}")

        return "\n".join(analysis)

    def analyze_prompt_quality(self, prompts: Dict[str, str]) -> str:
        """Analyze quality and completeness of generated prompts."""
        if not prompts:
            return "No prompts generated"

        analysis = ["Prompt Quality Analysis:"]
        analysis.append(f"Total prompts generated: {len(prompts)}")

        quality_indicators = {
            "technical_specs": 0,
            "implementation_checklist": 0,
            "success_criteria": 0,
            "dependencies": 0,
            "files_to_modify": 0
        }

        for task_id, prompt in prompts.items():
            prompt_lower = prompt.lower()

            # Check for quality indicators
            if "technical specs" in prompt_lower or "## technical specs" in prompt_lower:
                quality_indicators["technical_specs"] += 1
            if "implementation checklist" in prompt_lower or "checklist" in prompt_lower:
                quality_indicators["implementation_checklist"] += 1
            if "success criteria" in prompt_lower:
                quality_indicators["success_criteria"] += 1
            if "dependencies" in prompt_lower or "depends on" in prompt_lower:
                quality_indicators["dependencies"] += 1
            if "files to modify" in prompt_lower or "files to read" in prompt_lower:
                quality_indicators["files_to_modify"] += 1

            analysis.append(f"Task {task_id}: {len(prompt)} chars")

        analysis.append("Quality indicators across all prompts:")
        for indicator, count in quality_indicators.items():
            analysis.append(f"  {indicator}: {count}/{len(prompts)} prompts")

        return "\n".join(analysis)

    def analyze_expected_vs_planned_dependencies(self, devai_requirements: List[Dict], prompts: Dict[str, str], planning_data: Dict) -> str:
        """Build expected dependency tree from DevAI requirements and compare with planned implementation."""
        analysis = ["Expected vs Planned Dependencies Analysis:"]

        # Build expected dependency graph from DevAI requirements
        expected_graph = self._build_expected_dependency_graph(devai_requirements)

        # Extract planned dependencies from conversation
        planned_dependencies = self._extract_planned_dependencies(prompts, planning_data)

        # Compare graphs
        comparison = self._compare_dependency_coverage(expected_graph, planned_dependencies)

        analysis.append(f"Expected files: {len(expected_graph['files'])}")
        analysis.append(f"Planned files: {len(planned_dependencies['files'])}")
        analysis.append(f"Coverage: {comparison['coverage_percentage']:.1f}%")

        if comparison['missing_files']:
            analysis.append("Missing required files:")
            for file_info in comparison['missing_files'][:5]:
                analysis.append(f"  - {file_info['path']} (req #{file_info['requirement_id']})")

        if comparison['extra_files']:
            analysis.append("Additional planned files:")
            for file_path in comparison['extra_files'][:5]:
                analysis.append(f"  + {file_path}")

        # Dependency relationship analysis
        if expected_graph['dependencies']:
            analysis.append(f"Expected dependency relationships: {len(expected_graph['dependencies'])}")
            dep_coverage = comparison['dependency_coverage']
            analysis.append(f"Dependency coverage: {dep_coverage:.1f}%")

        return "\n".join(analysis)

    def _build_expected_dependency_graph(self, devai_requirements: List[Dict]) -> Dict:
        """Build expected dependency graph from DevAI requirements."""
        import re

        expected_files = []
        dependencies = []

        for req in devai_requirements:
            criteria = req.get('criteria', '')
            req_id = req.get('requirement_id')
            prereqs = req.get('prerequisites', [])

            # Extract file paths from criteria using various patterns
            file_patterns = [
                r'`([^`]*\.py)`',           # Files in backticks
                r'in `([^`]*)`',            # "in `path`"
                r'as `([^`]*)`',            # "as `path`"
                r'([a-zA-Z_][a-zA-Z0-9_/]*\.py)',  # Direct .py files
                r'([a-zA-Z_][a-zA-Z0-9_/]*\.txt)',  # .txt files
                r'([a-zA-Z_][a-zA-Z0-9_/]*\.png)',  # Image files
            ]

            for pattern in file_patterns:
                matches = re.findall(pattern, criteria)
                for match in matches:
                    if match and '/' in match:  # Only structured paths
                        expected_files.append({
                            'path': match,
                            'requirement_id': req_id,
                            'category': req.get('category', 'Unknown'),
                            'prerequisites': prereqs
                        })

            # Build prerequisite dependencies
            for prereq in prereqs:
                dependencies.append({
                    'from_req': prereq,
                    'to_req': req_id,
                    'type': 'prerequisite'
                })

        return {
            'files': expected_files,
            'dependencies': dependencies,
            'requirements': devai_requirements
        }

    def _extract_planned_dependencies(self, prompts: Dict[str, str], planning_data: Dict) -> Dict:
        """Extract planned files and dependencies from conversation data."""
        import re

        planned_files = set()

        # Extract from prompts
        for task_id, prompt in prompts.items():
            file_patterns = [
                r'`([^`]*\.py)`',
                r'([a-zA-Z_][a-zA-Z0-9_/]*\.py)',
                r'([a-zA-Z_][a-zA-Z0-9_/]*\.txt)',
                r'([a-zA-Z_][a-zA-Z0-9_/]*\.png)',
            ]

            for pattern in file_patterns:
                matches = re.findall(pattern, prompt)
                for match in matches:
                    if match and ('/' in match or match.endswith(('.py', '.txt', '.png'))):
                        planned_files.add(match)

        # Extract from task descriptions in planning data
        plan_data = planning_data.get("plan", {})
        all_tasks = self.extract_all_tasks(plan_data)

        for task in all_tasks:
            task_text = f"{task.get('name', '')} {task.get('description', '')}"
            for pattern in [r'`([^`]*\.py)`', r'([a-zA-Z_][a-zA-Z0-9_/]*\.py)']:
                matches = re.findall(pattern, task_text)
                for match in matches:
                    if match and '/' in match:
                        planned_files.add(match)

        return {
            'files': list(planned_files),
            'task_count': len(all_tasks)
        }

    def _compare_dependency_coverage(self, expected_graph: Dict, planned_dependencies: Dict) -> Dict:
        """Compare expected vs planned dependency coverage."""
        expected_files = {f['path'] for f in expected_graph['files']}
        planned_files = set(planned_dependencies['files'])

        matched_files = expected_files.intersection(planned_files)
        missing_files = [f for f in expected_graph['files'] if f['path'] not in planned_files]
        extra_files = planned_files - expected_files

        coverage_percentage = (len(matched_files) / len(expected_files) * 100) if expected_files else 0

        # Dependency relationship coverage
        expected_deps = len(expected_graph['dependencies'])
        # For now, assume good coverage if files are covered
        dependency_coverage = coverage_percentage if expected_deps > 0 else 100

        return {
            'coverage_percentage': coverage_percentage,
            'matched_files': list(matched_files),
            'missing_files': missing_files,
            'extra_files': list(extra_files),
            'dependency_coverage': dependency_coverage
        }

    def analyze_action_coverage_vs_reference(self, devai_requirements: List[Dict], planning_data: Dict, planning_module) -> str:
        """Compare agent's planned actions against a reference plan generated for the DevAI task."""
        analysis = ["Action Coverage vs Reference Plan:"]

        try:
            # Generate reference plan from DevAI task
            devai_query = self._extract_devai_query(devai_requirements)
            reference_plan = planning_module.generate_plan(devai_query)
            reference_actions = reference_plan.get('actions', [])

            # Extract agent's actual planning actions from conversation
            agent_actions = self._extract_agent_actions_from_plan(planning_data)

            # Compare action coverage
            coverage_analysis = self._compare_action_coverage(agent_actions, reference_actions)

            analysis.append(f"Reference plan actions: {len(reference_actions)}")
            analysis.append(f"Agent planned actions: {len(agent_actions)}")
            analysis.append(f"Action coverage: {coverage_analysis['coverage_percentage']:.1f}%")

            if reference_actions:
                analysis.append("Reference action sequence:")
                for i, action in enumerate(reference_actions[:5], 1):
                    analysis.append(f"  {i}. {action}")
                if len(reference_actions) > 5:
                    analysis.append(f"  ... and {len(reference_actions) - 5} more actions")

            if coverage_analysis['missing_actions']:
                analysis.append("Missing actions in agent's plan:")
                for action in coverage_analysis['missing_actions'][:3]:
                    analysis.append(f"  - {action}")

            if coverage_analysis['extra_actions']:
                analysis.append("Additional actions in agent's plan:")
                for action in coverage_analysis['extra_actions'][:3]:
                    analysis.append(f"  + {action}")

            # Execution sequence analysis
            execution_quality = self._analyze_execution_sequence(planning_data)
            analysis.append(f"Execution sequence quality: {execution_quality['score']:.1f}/10")
            if execution_quality['issues']:
                analysis.append("Sequence issues:")
                for issue in execution_quality['issues'][:3]:
                    analysis.append(f"  - {issue}")

        except Exception as e:
            analysis.append(f"Reference plan generation failed: {str(e)}")
            # Fallback to basic execution analysis
            fallback_analysis = self._basic_execution_analysis(planning_data)
            analysis.extend(fallback_analysis)

        return "\n".join(analysis)

    def _extract_devai_query(self, devai_requirements: List[Dict]) -> str:
        """Extract the main task description from DevAI requirements to generate reference plan."""
        # Look for the original query if available
        if hasattr(self, 'devai_data') and 'query' in self.devai_data:
            return self.devai_data['query']

        # Otherwise, build a summary from requirements
        task_summary = "Develop a solution that:"
        for req in devai_requirements[:5]:  # Use first 5 requirements
            criteria = req.get('criteria', '')
            if criteria:
                task_summary += f" {criteria}."

        return task_summary

    def _extract_agent_actions_from_plan(self, planning_data: Dict) -> List[str]:
        """Extract the types of actions the agent planned to take."""
        actions = []

        # Check if plan includes typical agent actions
        plan_text = str(planning_data).lower()

        action_indicators = {
            'user_query': ['user', 'query', 'requirement'],
            'workspace': ['workspace', 'directory', 'structure'],
            'locate': ['locate', 'find', 'search for'],
            'read': ['read', 'examine', 'analyze'],
            'search': ['search', 'grep', 'find'],
            'history': ['history', 'previous', 'context']
        }

        for action_type, keywords in action_indicators.items():
            if any(keyword in plan_text for keyword in keywords):
                actions.append(action_type)

        # Extract from prompts if available
        prompts = planning_data.get('prompts', {})
        for prompt in prompts.values():
            prompt_lower = prompt.lower()
            for action_type, keywords in action_indicators.items():
                if any(keyword in prompt_lower for keyword in keywords):
                    if action_type not in actions:
                        actions.append(action_type)

        return actions

    def _compare_action_coverage(self, agent_actions: List[str], reference_actions: List[str]) -> Dict:
        """Compare agent actions against reference actions."""
        agent_set = set(agent_actions)
        reference_set = set(reference_actions)

        matched_actions = agent_set.intersection(reference_set)
        missing_actions = reference_set - agent_set
        extra_actions = agent_set - reference_set

        coverage_percentage = (len(matched_actions) / len(reference_set) * 100) if reference_set else 0

        return {
            'coverage_percentage': coverage_percentage,
            'matched_actions': list(matched_actions),
            'missing_actions': list(missing_actions),
            'extra_actions': list(extra_actions)
        }

    def _analyze_execution_sequence(self, planning_data: Dict) -> Dict:
        """Analyze the quality of the execution sequence."""
        score = 5.0  # Start with baseline
        issues = []

        order = planning_data.get('order', [])
        plan_data = planning_data.get('plan', {})

        if not order:
            return {'score': 0, 'issues': ['No execution order provided']}

        # Build task lookup
        task_lookup = {}
        def build_lookup(node: Dict):
            task_lookup[node.get('id')] = node
            for child in node.get('children', []):
                build_lookup(child)

        for root_node in plan_data.get("root", []):
            build_lookup(root_node)

        # Check for dependency violations
        violation_count = 0
        for i, task_id in enumerate(order):
            if task_id in task_lookup:
                task = task_lookup[task_id]
                depends_on_str = task.get('depends_on', 'set()')
                if depends_on_str not in ['set()', 'None', '']:
                    # Parse dependencies (simplified)
                    try:
                        deps = eval(depends_on_str) if depends_on_str.startswith('{') else set()
                        for dep in deps:
                            if dep in order:
                                dep_index = order.index(dep)
                                if dep_index > i:  # Dependency comes after dependent
                                    violation_count += 1
                                    issues.append(f"Task {task_id} depends on {dep} which comes later")
                    except:
                        pass

        # Adjust score based on violations
        if violation_count > 0:
            score -= min(3.0, violation_count * 0.5)
            issues.append(f"{violation_count} dependency violations found")

        # Check execution completeness
        if len(order) < 5:
            score -= 1.0
            issues.append("Execution plan seems too simple")
        elif len(order) > 50:
            score -= 0.5
            issues.append("Execution plan might be too complex")

        return {'score': max(0, score), 'issues': issues}

    def _basic_execution_analysis(self, planning_data: Dict) -> List[str]:
        """Fallback execution analysis when reference plan generation fails."""
        analysis = []
        order = planning_data.get('order', [])
        plan_data = planning_data.get('plan', {})

        analysis.append(f"Basic execution analysis:")
        analysis.append(f"Total execution steps: {len(order)}")

        if order:
            # Build task lookup for basic stats
            task_lookup = {}
            def build_lookup(node: Dict):
                task_lookup[node.get('id')] = node
                for child in node.get('children', []):
                    build_lookup(child)

            for root_node in plan_data.get("root", []):
                build_lookup(root_node)

            # Group by level
            task_levels = {}
            for task_id in order:
                if task_id in task_lookup:
                    level = task_lookup[task_id].get('level', 'unknown')
                    task_levels[level] = task_levels.get(level, 0) + 1

            analysis.append("Task distribution:")
            for level, count in task_levels.items():
                analysis.append(f"  - {level}: {count} tasks")

        return analysis


class DependencyGraphValidator:
    """Validates planning dependencies using NetworkX graph algorithms."""

    def __init__(self):
        self.planning_graph = None
        self.devai_graph = None

    def parse_depends_on(self, depends_on_str: str) -> Set[int]:
        """Parse depends_on string representation to set of task IDs."""
        if not depends_on_str or depends_on_str in ('set()', 'None'):
            return set()

        # Handle string representations like "{2}", "{3,4}", etc.
        if depends_on_str.startswith('{') and depends_on_str.endswith('}'):
            content = depends_on_str[1:-1].strip()
            if not content:
                return set()
            try:
                return {int(x.strip()) for x in content.split(',') if x.strip()}
            except ValueError:
                logging.warning(f"Failed to parse depends_on: {depends_on_str}")
                return set()

        return set()

    def extract_all_tasks(self, plan_data: Dict) -> List[Dict]:
        """Extract all tasks from hierarchical plan structure."""
        tasks = []

        def extract_recursive(node: Dict):
            tasks.append(node)
            for child in node.get('children', []):
                extract_recursive(child)

        for root_node in plan_data.get("root", []):
            extract_recursive(root_node)

        return tasks

    def build_planning_dependency_graph(self, planning_data: Dict) -> nx.DiGraph:
        """Build NetworkX graph from planning conversation results."""
        G = nx.DiGraph()

        plan_data = planning_data.get("plan", {})
        if not plan_data:
            return G

        # Extract all tasks
        all_tasks = self.extract_all_tasks(plan_data)

        # Add nodes for each task
        for task in all_tasks:
            task_id = task.get('id')
            if task_id is not None:
                G.add_node(task_id,
                          name=task.get('name', ''),
                          description=task.get('description', ''),
                          level=task.get('level', ''))

        # Add dependency edges
        for task in all_tasks:
            task_id = task.get('id')
            if task_id is None:
                continue

            depends_on_str = task.get('depends_on', 'set()')
            dependencies = self.parse_depends_on(depends_on_str)

            for dep_id in dependencies:
                if dep_id in G.nodes:
                    G.add_edge(dep_id, task_id)
                else:
                    logging.warning(f"Task {task_id} depends on non-existent task {dep_id}")

        self.planning_graph = G
        return G

    def build_devai_dependency_graph(self, devai_requirements: List[Dict]) -> nx.DiGraph:
        """Build NetworkX graph from DevAI requirements prerequisites."""
        G = nx.DiGraph()

        # Add nodes for each requirement
        for req in devai_requirements:
            req_id = req.get('requirement_id')
            if req_id is not None:
                G.add_node(req_id,
                          criteria=req.get('criteria', ''),
                          category=req.get('category', ''))

        # Add dependency edges from prerequisites
        for req in devai_requirements:
            req_id = req.get('requirement_id')
            if req_id is None:
                continue

            prerequisites = req.get('prerequisites', [])
            for prereq_id in prerequisites:
                if prereq_id in G.nodes:
                    G.add_edge(prereq_id, req_id)
                else:
                    logging.warning(f"Requirement {req_id} has invalid prerequisite {prereq_id}")

        self.devai_graph = G
        return G

    def validate_dependency_ordering(self, planning_data: Dict, devai_requirements: List[Dict]) -> Dict[str, Any]:
        """Validate that planned dependencies satisfy DevAI requirements."""

        # Build both graphs
        plan_graph = self.build_planning_dependency_graph(planning_data)
        devai_graph = self.build_devai_dependency_graph(devai_requirements)

        validation_results = {
            "plan_tasks": len(plan_graph.nodes),
            "devai_requirements": len(devai_graph.nodes),
            "dependency_violations": [],
            "topological_valid": True,
            "cycle_detection": {},
            "coverage_analysis": {}
        }

        # Check for cycles in planning graph
        try:
            plan_order = list(nx.topological_sort(plan_graph))
            validation_results["plan_topological_order"] = plan_order
        except nx.NetworkXError as e:
            validation_results["topological_valid"] = False
            validation_results["cycle_detection"]["planning"] = str(e)
            # Find cycles
            try:
                cycles = list(nx.simple_cycles(plan_graph))
                validation_results["cycle_detection"]["cycles"] = cycles
            except:
                pass

        # Check for cycles in DevAI requirements
        try:
            devai_order = list(nx.topological_sort(devai_graph))
            validation_results["devai_topological_order"] = devai_order
        except nx.NetworkXError as e:
            validation_results["cycle_detection"]["devai"] = str(e)

        # Validate specific dependency constraints
        violations = []

        # For each DevAI requirement, check if planning satisfies its dependencies
        task_to_requirement_mapping = self._map_tasks_to_requirements(
            planning_data, devai_requirements
        )

        for req in devai_requirements:
            req_id = req.get('requirement_id')
            prerequisites = req.get('prerequisites', [])

            if not prerequisites:
                continue

            # Find planning task(s) that address this requirement
            addressing_tasks = task_to_requirement_mapping.get(req_id, [])

            if not addressing_tasks:
                violations.append({
                    "requirement_id": req_id,
                    "type": "missing_task",
                    "message": f"No planning task found for requirement {req_id}"
                })
                continue

            # Check if prerequisites are satisfied for each addressing task
            for task_id in addressing_tasks:
                if task_id not in plan_graph.nodes:
                    continue

                # Get all predecessors of this task in planning graph
                predecessors = set(nx.ancestors(plan_graph, task_id))

                # Check if all required prerequisites are covered
                for prereq_req_id in prerequisites:
                    prereq_tasks = task_to_requirement_mapping.get(prereq_req_id, [])

                    if not prereq_tasks:
                        violations.append({
                            "requirement_id": req_id,
                            "task_id": task_id,
                            "type": "missing_prerequisite_task",
                            "prerequisite_req_id": prereq_req_id,
                            "message": f"Task {task_id} needs prerequisite req {prereq_req_id} but no task found"
                        })
                        continue

                    # Check if any prerequisite task is a predecessor
                    prereq_satisfied = any(prereq_task in predecessors for prereq_task in prereq_tasks)

                    if not prereq_satisfied:
                        violations.append({
                            "requirement_id": req_id,
                            "task_id": task_id,
                            "type": "dependency_ordering",
                            "prerequisite_req_id": prereq_req_id,
                            "prerequisite_tasks": prereq_tasks,
                            "message": f"Task {task_id} should depend on tasks {prereq_tasks} but doesn't"
                        })

        validation_results["dependency_violations"] = violations
        validation_results["violation_count"] = len(violations)

        # Coverage analysis
        covered_requirements = set(task_to_requirement_mapping.keys())
        all_requirements = {req['requirement_id'] for req in devai_requirements}
        missing_requirements = all_requirements - covered_requirements

        validation_results["coverage_analysis"] = {
            "total_requirements": len(all_requirements),
            "covered_requirements": len(covered_requirements),
            "missing_requirements": list(missing_requirements),
            "coverage_percentage": len(covered_requirements) / len(all_requirements) * 100 if all_requirements else 0
        }

        return validation_results

    def _map_tasks_to_requirements(self, planning_data: Dict, devai_requirements: List[Dict]) -> Dict[int, List[int]]:
        """Map DevAI requirements to planning tasks based on content similarity."""
        mapping = {}

        # Extract all planning tasks
        plan_data = planning_data.get("plan", {})
        all_tasks = self.extract_all_tasks(plan_data)

        # Simple keyword-based mapping (could be enhanced with embeddings)
        for req in devai_requirements:
            req_id = req.get('requirement_id')
            req_criteria = req.get('criteria', '').lower()

            # Extract key terms from requirement
            key_terms = self._extract_key_terms(req_criteria)

            matching_tasks = []
            for task in all_tasks:
                task_id = task.get('id')
                if task_id is None:
                    continue

                task_text = f"{task.get('name', '')} {task.get('description', '')}".lower()

                # Check if task addresses this requirement
                if self._text_similarity_score(key_terms, task_text) > 0.3:
                    matching_tasks.append(task_id)

            if matching_tasks:
                mapping[req_id] = matching_tasks

        return mapping

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from requirement criteria."""
        # Remove common technical terms and focus on domain-specific terms
        text = re.sub(r'is (implemented|loaded|saved|used|generated) in', '', text)
        text = re.sub(r'\.py|\.txt|\.png|\.html|\.csv', '', text)
        text = re.sub(r'`[^`]*`', '', text)  # Remove code snippets

        # Extract meaningful terms (could be enhanced with NLP)
        terms = []
        words = text.split()
        for word in words:
            word = word.strip('.,():[]{}')
            if len(word) > 3 and word.isalpha():
                terms.append(word)

        return terms

    def _text_similarity_score(self, key_terms: List[str], task_text: str) -> float:
        """Calculate similarity score between requirement terms and task text."""
        if not key_terms:
            return 0.0

        matches = sum(1 for term in key_terms if term in task_text)
        return matches / len(key_terms)

    def generate_dependency_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate human-readable dependency validation report."""
        report = ["=== Dependency Validation Report ===\n"]

        # Summary
        report.append(f"Planning Tasks: {validation_results['plan_tasks']}")
        report.append(f"DevAI Requirements: {validation_results['devai_requirements']}")
        report.append(f"Dependency Violations: {validation_results['violation_count']}")
        report.append(f"Topologically Valid: {validation_results['topological_valid']}")

        # Coverage analysis
        coverage = validation_results.get('coverage_analysis', {})
        report.append(f"Requirements Coverage: {coverage.get('coverage_percentage', 0):.1f}%")

        if missing := coverage.get('missing_requirements'):
            report.append(f"Missing Requirements: {missing}")

        # Violations details
        if violations := validation_results.get('dependency_violations'):
            report.append("\n=== Dependency Violations ===")
            for violation in violations:
                report.append(f"- {violation['type']}: {violation['message']}")

        # Cycle detection
        if cycle_info := validation_results.get('cycle_detection'):
            if cycle_info:
                report.append("\n=== Cycle Detection Issues ===")
                for key, value in cycle_info.items():
                    report.append(f"- {key}: {value}")

        return "\n".join(report)


class PlanningJudgeAgent:
    """Judge planning agent outputs against DevAI requirements."""

    def __init__(
        self,
        conversation_file: Path,
        devai_instance: Path,
        judge_dir: Path,
        config: Optional[AgentConfig] = None,
    ):
        self.conversation_file = conversation_file
        self.devai_instance = devai_instance
        self.judge_dir = judge_dir
        self.config = config or AgentConfig()

        import os
        model_name = os.getenv("DEFAULT_LLM", "claude-3-5-sonnet-20241022")

        # Use appropriate API key based on model
        if model_name.startswith("gpt-") or model_name.startswith("o1-"):
            api_key = os.getenv("OPENAI_API_KEY")
        elif model_name.startswith("claude-"):
            api_key = os.getenv("ANTHROPIC_API_KEY")
        else:
            # Try both, prefer OpenAI if both are available
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")

        self.llm = LLM(
            model=model_name,
            api_key=api_key
        )

        # Initialize components
        self.conversation_parser = ConversationParser(conversation_file)
        self.plan_analyzer = PlanAnalyzer()
        self.dependency_validator = DependencyGraphValidator()
        self.aaaj_ask = DevAsk(Path.cwd(), judge_dir)  # Workspace not used for planning
        self.planning_module = Planning()  # For reference plan generation

        # Load DevAI instance
        with open(devai_instance, 'r') as f:
            self.devai_data = json.load(f)

        # Statistics
        self.judge_stats = []
        self.total_time = 0.0

    def judge_planning_session(self) -> Dict[str, Any]:
        """Main method to judge the planning session."""
        logging.info(f"Judging planning session: {self.conversation_file.name}")

        start_time = time.time()

        # Extract planning data
        planning_data = self.conversation_parser.extract_planning_data()
        if not planning_data:
            return {
                "error": "Failed to parse conversation data",
                "conversation_file": str(self.conversation_file),
                "devai_instance": self.devai_data.get("name", "Unknown"),
                "session_complete": False,
                "total_requirements": 0,
                "satisfied_requirements": 0,
                "satisfaction_rate": 0,
                "total_time": time.time() - start_time,
                "judge_stats": []
            }

        # Validate dependencies first
        dependency_validation = self._validate_planning_dependencies(planning_data)

        # Judge each requirement - try both formats
        original_requirements = (
            self.devai_data.get("original_devai_requirements", []) or
            self.devai_data.get("requirements", [])
        )
        total_requirements = len(original_requirements)
        logging.info(f"Evaluating {total_requirements} requirements")

        for requirement in original_requirements:
            result = self._judge_requirement(requirement, planning_data, dependency_validation)
            self.judge_stats.append(result)

        self.total_time = time.time() - start_time

        # Generate summary
        satisfied_count = sum(1 for stat in self.judge_stats if stat.get("satisfied"))

        return {
            "conversation_file": str(self.conversation_file),
            "devai_instance": self.devai_data["name"],
            "session_complete": planning_data.get("session_complete", False),
            "total_requirements": total_requirements,
            "satisfied_requirements": satisfied_count,
            "satisfaction_rate": satisfied_count / total_requirements if total_requirements > 0 else 0,
            "total_time": self.total_time,
            "dependency_validation": dependency_validation,
            "judge_stats": self.judge_stats
        }

    def _judge_requirement(self, requirement: Dict, planning_data: Dict, dependency_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Judge a single DevAI requirement against planning output."""
        start_time = time.time()

        # Build evidence for this requirement
        evidence = self._build_planning_evidence(requirement, planning_data, dependency_validation)

        # Adapt requirement criteria for planning evaluation
        planning_criteria = self._adapt_criteria_for_planning(requirement["criteria"])

        # Use existing DevAsk to judge
        llm_stats = self.aaaj_ask.check(
            criteria=planning_criteria,
            evidence=evidence
        )

        result = {
            "requirement_id": requirement["requirement_id"],
            "original_criteria": requirement["criteria"],
            "planning_criteria": planning_criteria,
            "category": requirement["category"],
            "satisfied": llm_stats["satisfied"],
            "reason": llm_stats["reason"],
            "llm_stats": {
                "cost": llm_stats.get("cost", 0),
                "inference_time": llm_stats.get("inference_time", 0),
                "input_tokens": llm_stats.get("input_tokens", 0),
                "output_tokens": llm_stats.get("output_tokens", 0)
            },
            "evaluation_time": time.time() - start_time
        }

        logging.info(f"Requirement {requirement['requirement_id']}: {result['satisfied']}")
        return result

    def _validate_planning_dependencies(self, planning_data: Dict) -> Dict[str, Any]:
        """Validate planning dependencies against DevAI requirements."""
        original_requirements = (
            self.devai_data.get("original_devai_requirements", []) or
            self.devai_data.get("requirements", [])
        )
        validation_results = self.dependency_validator.validate_dependency_ordering(
            planning_data, original_requirements
        )

        # Generate report for logging
        report = self.dependency_validator.generate_dependency_report(validation_results)
        logging.info(f"Dependency Validation Results:\n{report}")

        return validation_results

    def _build_planning_evidence(self, requirement: Dict, planning_data: Dict, dependency_validation: Dict[str, Any]) -> str:
        """Build evidence string for judging a specific requirement."""
        evidence_parts = []

        # Add original task context
        if original_task := planning_data.get("original_task"):
            evidence_parts.append(f"Original Task:\n{original_task[:500]}...\n")

        # Add plan structure analysis
        plan_analysis = self.plan_analyzer.analyze_plan_structure(planning_data["plan"])
        evidence_parts.append(f"Plan Structure:\n{plan_analysis}\n")

        # Add dependency analysis
        deps_analysis = self.plan_analyzer.analyze_dependencies(
            planning_data["plan"],
            planning_data["order"]
        )
        evidence_parts.append(f"Dependencies:\n{deps_analysis}\n")

        # Add prompt quality analysis
        prompt_analysis = self.plan_analyzer.analyze_prompt_quality(planning_data["prompts"])
        evidence_parts.append(f"Prompts:\n{prompt_analysis}\n")

        # Add expected vs planned dependency analysis
        try:
            original_requirements = self.devai_data.get("requirements", [])
            dependency_analysis = self.plan_analyzer.analyze_expected_vs_planned_dependencies(
                original_requirements, planning_data["prompts"], planning_data
            )
            evidence_parts.append(f"Expected vs Planned Dependencies:\n{dependency_analysis}\n")
        except Exception as e:
            evidence_parts.append(f"Expected vs Planned Dependencies: Analysis failed ({str(e)})\n")

        # Add action coverage analysis against reference plan
        try:
            original_requirements = self.devai_data.get("requirements", [])
            action_analysis = self.plan_analyzer.analyze_action_coverage_vs_reference(
                original_requirements, planning_data, self.planning_module
            )
            evidence_parts.append(f"Action Coverage vs Reference:\n{action_analysis}\n")
        except Exception as e:
            evidence_parts.append(f"Action Coverage vs Reference: Analysis failed ({str(e)})\n")

        # Add dependency validation results for this specific requirement
        req_id = requirement.get('requirement_id')
        if req_id is not None:
            relevant_violations = [
                v for v in dependency_validation.get('dependency_violations', [])
                if v.get('requirement_id') == req_id
            ]
            if relevant_violations:
                evidence_parts.append(f"Dependency Violations for Requirement {req_id}:")
                for violation in relevant_violations:
                    evidence_parts.append(f"  - {violation['message']}")
                evidence_parts.append("")

        # Add overall dependency validation summary
        validation_summary = (
            f"Overall Dependency Validation:\n"
            f"  - Violations: {dependency_validation.get('violation_count', 0)}\n"
            f"  - Coverage: {dependency_validation.get('coverage_analysis', {}).get('coverage_percentage', 0):.1f}%\n"
            f"  - Topologically Valid: {dependency_validation.get('topological_valid', False)}\n"
        )
        evidence_parts.append(validation_summary)

        # Add session completion status
        evidence_parts.append(f"Session Complete: {planning_data.get('session_complete', False)}")

        evidence = "\n".join(evidence_parts)
        return truncate_string(evidence, model=self.llm.model_name, max_tokens=8000)

    def _adapt_criteria_for_planning(self, original_criteria: str) -> str:
        """Convert implementation criteria to planning criteria."""
        # Transform "X is implemented in file.py" to "Plan includes task to implement X in file.py"
        adaptations = [
            ("is loaded in", "plan includes task to load data in"),
            ("is implemented in", "plan includes task to implement"),
            ("is used", "plan includes task to use"),
            ("is saved", "plan includes task to save"),
            ("saved as", "plan includes task to save results as"),
            ("generated", "plan includes task to generate"),
            ("containing", "plan includes task to create content containing"),
        ]

        planning_criteria = original_criteria
        for old, new in adaptations:
            planning_criteria = planning_criteria.replace(old, new)

        # Add planning context
        planning_criteria = f"Based on the planning session evidence, evaluate: {planning_criteria}"

        return planning_criteria

    def save_judgment(self, output_file: Path):
        """Save judgment results to file."""
        results = self.judge_planning_session()

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

        logging.info(f"Judgment saved to {output_file}")
        return results