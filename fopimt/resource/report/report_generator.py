from pathlib import Path
from typing import ClassVar

from fopimt.resource.report.report_dto import ModulVisualizationDto
from fopimt.resource.report.report_exporters import (
    BaseReportExporter,
    HtmlReportExporter,
    LatexReportExporter,
)
from fopimt.solutions.solution import Solution
from fopimt.task_dto import TaskExecutionContext

from .report_data_builders import (
    SolutionReportDataBuilder,
    TaskReportDataBuilder,
)
from .report_dto import OutputFormat


class ReportGenerator:
    _SOLUTION_TEMPLATE_NAME: ClassVar[str] = "template_solution.jinja2"
    _TASK_TEMPLATE_NAME: ClassVar[str] = "template_task.jinja2"

    def __init__(self, task_execution_context: TaskExecutionContext) -> None:
        self._task_context: TaskExecutionContext = task_execution_context
        self._resource_root_dir_path = Path(__file__).resolve().parent

    @property
    def task_context(self) -> TaskExecutionContext:
        if self._task_context is None:
            raise RuntimeError("Task context is not set")
        return self._task_context

    @task_context.setter
    def task_context(self, value: TaskExecutionContext) -> None:
        self._task_context = value

    def _validate_common_export_state(self, output_format: OutputFormat) -> None:
        if output_format not in OutputFormat:
            raise ValueError(f"Unsupported output format: {output_format}")

        if self._task_context is None:
            raise RuntimeError("Task context must be set before exporting report")

    def _validate_solution_export_state(
        self,
        output_format: OutputFormat,
        solution: Solution | None,
        evaluator_visualization: ModulVisualizationDto | None,
    ) -> None:
        self._validate_common_export_state(output_format)

        if solution is None:
            raise ValueError("Solution must be provided for solution report type")

        if evaluator_visualization is None:
            raise ValueError(
                "Evaluator visualization must be provided for solution report type"
            )

    def _validate_task_export_state(self, output_format: OutputFormat) -> None:
        self._validate_common_export_state(output_format)

    def _get_exporter(self, output_format: OutputFormat) -> BaseReportExporter:
        if output_format == OutputFormat.HTML:
            return HtmlReportExporter(self._resource_root_dir_path)

        if output_format == OutputFormat.LATEX:
            return LatexReportExporter(self._resource_root_dir_path)

        raise ValueError(f"Unsupported output format: {output_format}")

    def export_solution_report(
        self,
        output_format: OutputFormat,
        solution: Solution,
        evaluator_visualization: ModulVisualizationDto,
        out_dir_path: str,
    ) -> None:
        self._validate_solution_export_state(
            output_format=output_format,
            solution=solution,
            evaluator_visualization=evaluator_visualization,
        )

        builder = SolutionReportDataBuilder(
            task_context=self._task_context,
            resource_root_dir_path=self._resource_root_dir_path,
        )
        report_data = builder.build(evaluator_visualization, solution, output_format)

        exporter = self._get_exporter(output_format)
        exporter.export(
            report_data=report_data,
            out_dir_path=out_dir_path,
            template_name=self._SOLUTION_TEMPLATE_NAME,
        )

    def export_task_report(
        self,
        output_format: OutputFormat,
        modules_visualizations_by_iteration: dict[int, list[ModulVisualizationDto]],
        stats_visualizations: list[ModulVisualizationDto],
        out_dir_path: str,
    ) -> None:
        self._validate_task_export_state(output_format=output_format)

        builder = TaskReportDataBuilder(
            task_context=self._task_context,
            resource_root_dir_path=self._resource_root_dir_path,
        )
        report_data = builder.build(
            modules_visualizations_by_iteration,
            stats_visualizations,
            output_format,
        )

        exporter = self._get_exporter(output_format)
        exporter.export(
            report_data=report_data,
            out_dir_path=out_dir_path,
            template_name=self._TASK_TEMPLATE_NAME,
        )
