import base64
import json
import logging
import mimetypes
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import markdown

from fopimt.loader_dto import PackageType, Parameter
from fopimt.message import Message
from fopimt.modul import Modul
from fopimt.resource.report.report_dto import ModulVisualizationDto, OutputFormat
from fopimt.solutions.solution import Solution
from fopimt.task_dto import OptimizationGoal, TaskExecutionContext


@dataclass(frozen=True)
class SolutionReportData:
    page_title: str = "EASE Solution Report"
    logo_data_uri: str = ""
    generated_at: str = "—"
    task_name: str = "Unnamed task"
    task_id: str = "task_unknown"
    solution_number: int = 1
    fitness: float | str = "—"
    fitness_note: str = "Fitness computed by task-specific evaluator"
    solver_name: str = "unknown"
    evaluator_name: str = "unknown"
    runtime: str = "—"
    solution_title: str = "generated solution"
    viz_title: str = "Result summary"
    viz_module_name: str = "unknown"
    solution_open: bool = False
    solution_content: str = ""
    custom_viz_rendered: str = ""
    theme: str = "dark"
    raw_metadata_json: str = "{}"
    raw_metadata_open: bool = False


@dataclass
class TaskVizItem:
    title: str
    module: str
    rendered_content: str | None = None


@dataclass(frozen=True)
class IterationColumns:
    iteration: list[int] = field(default_factory=list)
    fitness: list[float | None] = field(default_factory=list)
    avg_fitness: list[float | None] = field(default_factory=list)
    fitness_convergence: list[float | None] = field(default_factory=list)

    def __post_init__(self) -> None:
        lengths = {
            len(self.iteration),
            len(self.fitness),
            len(self.avg_fitness),
            len(self.fitness_convergence),
        }

        if len(lengths) > 1:
            raise ValueError("All iteration series arrays must have the same length.")


@dataclass(frozen=True)
class IterationSeries:
    columns: IterationColumns = field(default_factory=IterationColumns)


@dataclass(frozen=True)
class ModulInfoViz:
    modul_name: str
    parameters_json: str


@dataclass(frozen=True)
class TaskReportData:
    # --------------------------------------------------
    # General / Branding
    # --------------------------------------------------
    page_title: str = "EASE Task Report"
    logo_data_uri: str = ""
    generated_at: str = "—"
    theme: str = "dark"

    # --------------------------------------------------
    # Task Identity
    # --------------------------------------------------
    task_name: str = "Unnamed task"
    task_id: str = "task_unknown"
    task_author: str = ""

    # --------------------------------------------------
    # High-Level Fitness Summary
    # --------------------------------------------------
    best_fitness: Union[float, str] = "—"
    best_fitness_note: str = "Computed by task-specific evaluator"

    # --------------------------------------------------
    # Runtime / Summary Metrics
    # --------------------------------------------------
    time_start: str = "—"
    total_runtime: str = "—"
    tokens_used: int = 0
    valid_iterations: int = 0
    invalid_iterations: int = 0
    best_solution_number: Union[int, str] = "—"

    # --------------------------------------------------
    # Messages / Prompt Content
    # --------------------------------------------------
    system_message: str = ""
    initial_message: str = ""
    system_message_rendered: str = ""
    initial_message_rendered: str = ""
    repeated_message_type: str | None = None
    repeated_message_messages: list[str] = field(default_factory=list)
    repeated_message_weights: list[str] = field(default_factory=list)

    # --------------------------------------------------
    # Used Modules
    # --------------------------------------------------
    used_modules: list[ModulInfoViz] = field(default_factory=list)
    evaluator_name: str = "unknown"
    solution_name: str = "unknown"

    # --------------------------------------------------
    # Iteration Graph Data
    # --------------------------------------------------
    iteration_series: IterationSeries = field(default_factory=IterationSeries)

    # --------------------------------------------------
    # Best Solution Snapshot
    # --------------------------------------------------
    best_solution_id: str = ""
    best_runtime: str = "—"
    best_solution_preview: str = ""

    # --------------------------------------------------
    # Module Visualizations
    # --------------------------------------------------
    viz_items_by_iteration: dict[int, list[TaskVizItem]] = field(default_factory=dict)
    stats_viz_items: list[TaskVizItem] = field(default_factory=list)


class BaseReportDataBuilder:
    def __init__(
        self,
        task_context: TaskExecutionContext,
        resource_root_dir_path: Path,
    ) -> None:
        self._task_context = task_context
        self._resource_root_dir_path = resource_root_dir_path

    def _get_solver(self) -> type[Modul]:
        solvers = [
            m.class_ref
            for m in self._task_context.used_modules
            if m.package_type == PackageType.Solution
        ]
        if len(solvers) != 1:
            logging.error(f"Solvers: {solvers}")
            raise ValueError(f"Expected exactly 1 Solution, got {len(solvers)}")
        return solvers[0]

    def _get_evaluator(self) -> type[Modul]:
        evaluators = [
            m.class_ref
            for m in self._task_context.used_modules
            if m.package_type == PackageType.Evaluator
        ]
        if len(evaluators) != 1:
            logging.error(f"Evaluators: {evaluators}")
            raise ValueError(f"Expected exactly 1 Evaluator, got {len(evaluators)}")
        return evaluators[0]

    def _load_logo_data_uri(self) -> str:
        logo_path = self._resource_root_dir_path.parents[2] / "static" / "ease-logo.png"
        if logo_path.exists():
            mime_type = mimetypes.guess_type(str(logo_path))[0] or "image/png"
            with open(logo_path, "rb") as logo_file:
                encoded = base64.b64encode(logo_file.read()).decode("ascii")
                return f"data:{mime_type};base64,{encoded}"
        logging.warning(f"Logo file not found at expected path: {logo_path}")
        return ""

    def _calculate_runtime_ms(
        self, time_start: datetime | None, time_end: datetime | None
    ) -> int | str:
        if time_start is None or time_end is None:
            return "—"

        try:
            start = (
                time_start.astimezone(timezone.utc).replace(tzinfo=None)
                if time_start.tzinfo is not None
                else time_start
            )
            end = (
                time_end.astimezone(timezone.utc).replace(tzinfo=None)
                if time_end.tzinfo is not None
                else time_end
            )
            runtime_ms = int((end - start).total_seconds() * 1000)
            return runtime_ms if runtime_ms >= 0 else "—"
        except Exception:
            return "—"

    def _format_duration_hhmmss_ms(self, duration_ms: int | str) -> str:
        if not isinstance(duration_ms, int) or duration_ms < 0:
            return "—"

        hours, remainder = divmod(duration_ms, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        seconds, milliseconds = divmod(remainder, 1_000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    def _calculate_runtime_formatted(
        self, time_start: datetime | None, time_end: datetime | None
    ) -> str:
        runtime_ms = self._calculate_runtime_ms(time_start, time_end)
        return self._format_duration_hhmmss_ms(runtime_ms)

    def _markdown_to_html(self, md: str) -> str:
        if not md:
            return ""

        extensions = [
            "extra",
            "toc",
        ]
        return markdown.markdown(md, extensions=extensions)

    def _message_to_raw_text(self, message: Message | None) -> str:
        if not message:
            return ""
        return message.get_content() or ""

    def _message_to_rendered_html(self, message: Message | None) -> str:
        content = self._message_to_raw_text(message)
        return self._markdown_to_html(content) if content else ""

    def _get_optimization_goal(self) -> OptimizationGoal:
        optimization_goal = self._task_context.optimization_goal
        if optimization_goal is None:
            logging.warning(
                "No optimization goal specified in task context, defaulting to minimization."
            )
            return OptimizationGoal.MINIMIZATION
        return optimization_goal

    def _get_optimization_goal_note(self) -> str:
        optimization_goal = self._get_optimization_goal()
        if optimization_goal == OptimizationGoal.MINIMIZATION:
            return "Lower fitness is better (minimization)."
        return "Higher fitness is better (maximization)."


class SolutionReportDataBuilder(BaseReportDataBuilder):
    def build(
        self,
        evaluator_visualization: ModulVisualizationDto,
        solution: Solution,
        output_format: OutputFormat,
    ) -> SolutionReportData:
        if self._task_context is None:
            raise RuntimeError("Task context must be set for solution report")

        if (
            not evaluator_visualization
            or evaluator_visualization.package_type != PackageType.Evaluator
        ):
            raise ValueError(
                "Expected exactly one module visualization for the evaluator module"
            )

        evaluator_visualization_rendered = (
            evaluator_visualization.visualization_content or ""
        )

        time_start = solution.get_time_start() if solution else None
        time_end = solution.get_time_end() if solution else None
        runtime = self._calculate_runtime_formatted(time_start, time_end)

        if solution:
            solution_content = solution.get_input() or ""
            metadata = solution.get_metadata() or {}
            fitness = solution.get_fitness()
            if fitness is None:
                fitness = "—"
        else:
            solution_content = ""
            metadata = {}
            fitness = "—"

        raw_metadata_json = json.dumps(
            metadata, indent=2, ensure_ascii=False, default=str
        )

        logo_data_uri = self._load_logo_data_uri()

        return SolutionReportData(
            page_title=f"EASE Solution Report - #{self._task_context.current_iteration}",
            logo_data_uri=logo_data_uri,
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z"),
            task_name=self._task_context.task_name,
            task_id=self._task_context.task_id,
            solution_number=self._task_context.current_iteration,
            fitness=fitness,
            fitness_note=self._get_optimization_goal_note(),
            solver_name=self._get_solver().get_short_name(),
            evaluator_name=self._get_evaluator().get_short_name(),
            runtime=runtime,
            viz_title=f"{self._get_evaluator().get_long_name()}",
            viz_module_name=f"{self._get_evaluator().get_short_name()}",
            solution_title=f"solution #{self._task_context.current_iteration}",
            solution_open=bool(solution_content),
            solution_content=solution_content,
            custom_viz_rendered=evaluator_visualization_rendered,
            raw_metadata_json=raw_metadata_json,
            raw_metadata_open=bool(metadata),
        )


class TaskReportDataBuilder(BaseReportDataBuilder):
    def _get_single_module_name(self, package_type: PackageType) -> str | None:
        names = [
            m.class_ref.get_short_name()
            for m in self._task_context.used_modules
            if m.package_type == package_type
        ]
        if len(names) == 1:
            return names[0]
        return None

    def _get_time_start_str(self) -> str:
        time_start = self._task_context.time_start
        return time_start.strftime("%Y-%m-%d %H:%M:%S %Z") if time_start else "—"

    def _build_iteration_series(self, solutions: list) -> IterationSeries:
        fitness_values: list[float | None] = [
            solution.get_fitness() for solution in solutions
        ]
        iteration_values = list(range(len(fitness_values)))

        avg_fitness_values: list[float | None] = []
        fitness_convergence_values: list[float | None] = []
        running_sum = 0.0
        valid_count = 0
        best_so_far: float | None = None
        optimization_goal = self._get_optimization_goal()
        select_best = min if optimization_goal == OptimizationGoal.MINIMIZATION else max

        for fitness in fitness_values:
            if fitness is not None:
                running_sum += fitness
                valid_count += 1
                best_so_far = (
                    fitness if best_so_far is None else select_best(best_so_far, fitness)
                )

            avg_fitness_values.append(
                running_sum / valid_count if fitness is not None else None
            )
            fitness_convergence_values.append(best_so_far)

        return IterationSeries(
            columns=IterationColumns(
                iteration=iteration_values,
                fitness=fitness_values,
                avg_fitness=avg_fitness_values,
                fitness_convergence=fitness_convergence_values,
            )
        )

    def _get_best_solution_stats(
        self,
        solutions: list,
    ) -> tuple[float | None, object | None, int | str, str, str, str]:
        optimization_goal = self._get_optimization_goal()

        solutions_with_fitness = []
        for index, solution in enumerate(solutions):
            fitness = solution.get_fitness()
            if fitness is not None:
                solutions_with_fitness.append((fitness, solution, index))

        select_best = min if optimization_goal == OptimizationGoal.MINIMIZATION else max

        best_fitness, best_solution, best_solution_number = select_best(
            solutions_with_fitness,
            key=lambda item: item[0],
            default=(None, None, "—"),
        )

        best_solution_id = ""
        best_solution_preview = ""
        best_runtime = "—"

        if best_solution is not None:
            best_solution_id = best_solution.get_id() or ""
            best_solution_preview = best_solution.get_input() or ""

            best_runtime = self._calculate_runtime_formatted(
                best_solution.get_time_start(),
                best_solution.get_time_end(),
            )

        return (
            best_fitness,
            best_solution,
            best_solution_number,
            best_solution_id,
            best_solution_preview,
            best_runtime,
        )

    def _build_task_viz_items(
        self,
        modules_visualizations_by_iteration: dict[int, list[ModulVisualizationDto]],
    ) -> dict[int, list[TaskVizItem]]:
        out: dict[int, list[TaskVizItem]] = {}

        for iteration, visualizations in (
            modules_visualizations_by_iteration or {}
        ).items():
            if not visualizations:
                continue

            items: list[TaskVizItem] = []
            for visualization in visualizations:
                if not visualization.visualization_content:
                    continue

                module_name = visualization.modul.get_long_name()
                module_class_name = f"{visualization.modul.__name__}"
                items.append(
                    TaskVizItem(
                        title=module_name,
                        module=module_class_name,
                        rendered_content=visualization.visualization_content,
                    )
                )

            if items:
                out[iteration] = items

        return out

    def _build_stat_viz_items(
        self,
        stats_visualizations: list[ModulVisualizationDto],
    ) -> list[TaskVizItem]:
        items: list[TaskVizItem] = []

        for visualization in stats_visualizations or []:
            if not visualization.visualization_content:
                continue

            module_name = visualization.modul.get_long_name()
            module_class_name = f"{visualization.modul.__name__}"
            items.append(
                TaskVizItem(
                    title=module_name,
                    module=module_class_name,
                    rendered_content=visualization.visualization_content,
                )
            )

        return items

    def _build_used_modules_info_viz_items(self) -> list[ModulInfoViz]:
        items: list[ModulInfoViz] = []

        for modul_info in self._task_context.used_modules:
            parameters = {}
            for key, param in modul_info.parameters.items():
                if not isinstance(param, Parameter):
                    logging.warning(
                        f"Expected parameter '{key}' of module '{modul_info.class_ref.get_short_name()}' to be of type Parameter, got {type(param)}. Skipping this parameter in report."
                    )
                    continue

                parameters[key] = param.model_dump()

                if param.sensitive:
                    parameters[key]["value"] = "REDACTED"

            parameters_json = json.dumps(
                parameters,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

            items.append(
                ModulInfoViz(
                    modul_name=modul_info.class_ref.get_short_name(),
                    parameters_json=parameters_json,
                )
            )

        return items

    def build(
        self,
        modules_visualizations_by_iteration: dict[int, list[ModulVisualizationDto]],
        stats_visualizations: list[ModulVisualizationDto],
        output_format: OutputFormat,
    ) -> TaskReportData:
        if self._task_context is None:
            raise RuntimeError("Task context must be set for task report")

        solutions = self._task_context.solutions or []
        time_start_str = self._get_time_start_str()
        iteration_series = self._build_iteration_series(solutions)

        (
            best_fitness_value,
            _best_solution,
            best_solution_number,
            best_solution_id,
            best_solution_preview,
            best_runtime,
        ) = self._get_best_solution_stats(solutions)

        used_modules_info_viz_items = self._build_used_modules_info_viz_items()
        evaluator_name = self._get_single_module_name(PackageType.Evaluator)
        solution_name = self._get_single_module_name(PackageType.Solution)

        viz_items_by_iteration = self._build_task_viz_items(
            modules_visualizations_by_iteration
        )
        stats_viz_items = self._build_stat_viz_items(stats_visualizations)

        total_runtime = self._calculate_runtime_formatted(
            self._task_context.time_start,
            datetime.now(timezone.utc),
        )

        system_message = self._message_to_raw_text(self._task_context.system_message)
        initial_message = self._message_to_raw_text(self._task_context.initial_message)

        system_message_rendered = ""
        initial_message_rendered = ""

        if output_format == OutputFormat.HTML:
            system_message_rendered = self._message_to_rendered_html(
                self._task_context.system_message
            )
            initial_message_rendered = self._message_to_rendered_html(
                self._task_context.initial_message
            )

        repeated_message_type = (
            self._task_context.repeating_message.type
            if self._task_context.repeating_message
            else None
        )
        repeated_message_type_str = (
            str(repeated_message_type) if repeated_message_type else None
        )

        repeated_message_message_or_messages = (
            self._task_context.repeating_message.msgs
            if self._task_context.repeating_message
            else []
        )
        repeated_message_messages = (
            list(repeated_message_message_or_messages)
            if repeated_message_message_or_messages
            else []
        )

        repeated_message_weights_tuple = (
            self._task_context.repeating_message.weights
            if self._task_context.repeating_message
            else tuple()
        )
        repeated_message_weights = (
            [
                f"{weight:.2f}" if isinstance(weight, float) else str(weight)
                for weight in repeated_message_weights_tuple
            ]
            if repeated_message_weights_tuple
            else []
        )

        return TaskReportData(
            logo_data_uri=self._load_logo_data_uri(),
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z"),
            task_name=self._task_context.task_name,
            task_id=self._task_context.task_id,
            time_start=time_start_str,
            best_fitness=best_fitness_value if best_fitness_value is not None else "—",
            best_fitness_note=self._get_optimization_goal_note(),
            used_modules=used_modules_info_viz_items,
            evaluator_name=evaluator_name or TaskReportData.evaluator_name,
            solution_name=solution_name or TaskReportData.solution_name,
            iteration_series=iteration_series,
            best_solution_number=best_solution_number,
            best_solution_id=best_solution_id,
            best_runtime=best_runtime,
            best_solution_preview=best_solution_preview,
            viz_items_by_iteration=viz_items_by_iteration,
            stats_viz_items=stats_viz_items,
            tokens_used=self._task_context.used_tokens,
            valid_iterations=self._task_context.valid_iterations,
            invalid_iterations=self._task_context.invalid_iterations,
            total_runtime=total_runtime,
            system_message=system_message,
            initial_message=initial_message,
            system_message_rendered=system_message_rendered,
            initial_message_rendered=initial_message_rendered,
            repeated_message_type=repeated_message_type_str,
            repeated_message_messages=repeated_message_messages,
            repeated_message_weights=repeated_message_weights,
        )
