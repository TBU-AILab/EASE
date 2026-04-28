from pathlib import Path

from fopimt.loader_dto import PackageType, Parameter, PrimitiveType
from fopimt.resource.report.report_dto import ModulVisualizationDto, OutputFormat

from ..resource.report.report_generator import ReportGenerator
from ..solutions.solution import Solution
from ..task_dto import TaskExecutionContext
from .analysis import Analysis, AnalysisResult


class AnalysisReportSolution(Analysis):
    _default_output_format = OutputFormat.HTML.value

    @classmethod
    def get_parameters(cls) -> dict[str, Parameter]:
        output_formats: list[str] = [member.value for member in OutputFormat]
        return {
            "output_format": Parameter(
                short_name="output_format",
                type=PrimitiveType.enum,
                long_name="Output Format",
                description="Format of the report output",
                enum_options=output_formats,
                default=cls._default_output_format,
            ),
        }

    def _init_params(self):
        super()._init_params()
        output_format = self.parameters.get(
            "output_format", self._default_output_format
        )

        valid_formats = [format.value for format in OutputFormat]
        if output_format not in valid_formats:
            raise ValueError(
                f"Invalid output format: {output_format}. Valid options are: {valid_formats}"
            )

        self.output_format: OutputFormat = OutputFormat(output_format)
        self._evaluator_class = None
        self._evaluator_result = None
        self._solution = None
        self._task_execution_context = None

    ####################################################################
    #########  Public functions
    ####################################################################
    def evaluate_analysis(
        self,
        solution: Solution,
        task_execution_context: TaskExecutionContext,
    ) -> AnalysisResult:
        current_iteration = max(task_execution_context.modules_data_by_iteration.keys())
        current_iteration_modules = (
            task_execution_context.modules_data_by_iteration.get(current_iteration, [])
        )

        evaluator_modules = [
            module
            for module in current_iteration_modules
            if module.package_type == PackageType.Evaluator
        ]

        if len(evaluator_modules) != 1:
            raise ValueError(
                "Expected exactly one evaluator module in task_modules with short_name starting "
                f"with 'eval', found {len(evaluator_modules)}: {[module.class_ref.get_short_name() for module in evaluator_modules]}"
            )

        self._evaluator_class = evaluator_modules[0].class_ref
        self._evaluator_result = evaluator_modules[0].result
        self._solution = solution
        self._task_execution_context = task_execution_context

        return AnalysisResult(
            class_ref=type(self),
        )

    def export(self, path: str, id: str) -> None:
        if self._evaluator_class is None or self._evaluator_result is None:
            raise RuntimeError(
                "Evaluator class and result must be set before exporting report"
            )

        if self._solution is None:
            raise RuntimeError("Solution must be set before exporting report")

        if self._task_execution_context is None:
            raise RuntimeError(
                "Task execution context must be set before exporting report"
            )

        report_root_path = Path(path) / id
        report_generator = ReportGenerator(self._task_execution_context)

        eval_visualization_content = self._render_module_visualization(
            module_class=self._evaluator_class,
            result=self._evaluator_result,
            task_execution_context=self._task_execution_context,
            out_dir_path=str(report_root_path),
        )

        eval_visualization = ModulVisualizationDto(
            modul=self._evaluator_class,
            package_type=PackageType.Evaluator,
            visualization_content=eval_visualization_content,
        )

        report_generator.export_solution_report(
            output_format=self.output_format,
            solution=self._solution,
            evaluator_visualization=eval_visualization,
            out_dir_path=str(report_root_path),
        )

    @classmethod
    def get_short_name(cls) -> str:
        return "anal.report_solution"

    @classmethod
    def get_long_name(cls) -> str:
        return "Solution Report"

    @classmethod
    def get_description(cls) -> str:
        return "Generates a report for the given solution."

    @classmethod
    def get_tags(cls) -> dict:
        return {"input": set(), "output": {"files"}}

    ####################################################################
    #########  Private functions
    ####################################################################

    def _render_module_visualization(
        self,
        module_class: type,
        result,
        task_execution_context: TaskExecutionContext,
        out_dir_path: str,
    ) -> str | None:
        render_method_name = self.output_format.render_method_name
        render_method = getattr(module_class, render_method_name, None)

        if not callable(render_method):
            available = sorted(
                name for name in dir(module_class) if name.startswith("render_")
            )
            raise NotImplementedError(
                f"{module_class.__name__} does not implement '{render_method_name}'. "
                f"Available: {available or 'none'}"
            )

        try:
            result = render_method(result, task_execution_context, out_dir_path)
        except Exception:
            print("Viz Failed")
            return None

        return result  # type: ignore
