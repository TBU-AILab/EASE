from fopimt.loader_dto import PackageType, Parameter, PrimitiveType
from fopimt.resource.report.report_dto import ModulVisualizationDto, OutputFormat
from fopimt.resource.report.report_generator import ReportGenerator
from fopimt.solutions.solution import Solution
from fopimt.stats.stat import Stat, StatResult
from fopimt.task_dto import TaskExecutionContext


class StatReportTask(Stat):
    """
    Statistical module for generating a report for the given task. The report includes visualizations of all modules
    and statistics executed during the task.
    """

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
        self._data = None
        self._task_execution_context: TaskExecutionContext | None = None

    ####################################################################
    #########  Public functions
    ####################################################################
    def evaluate_statistic(
        self,
        solutions: list[Solution],
        task_execution_context: TaskExecutionContext,
    ) -> StatResult:
        self._task_execution_context = task_execution_context

        return StatResult(class_ref=type(self))

    def export(self, path: str) -> None:
        if self._task_execution_context is None:
            raise RuntimeError("Task context must be set before exporting report")

        task_execution_context = self._task_execution_context

        modules_visualizations_by_iteration: dict[int, list[ModulVisualizationDto]] = {}
        stats_visualizations: list[ModulVisualizationDto] = []

        # Keep ordering stable and intuitive:
        # - iterations in ascending numeric order
        # - modules within each iteration in their original list order
        for iteration_index in sorted(
            task_execution_context.modules_data_by_iteration.keys()
        ):
            modules_data_for_iteration = (
                task_execution_context.modules_data_by_iteration.get(
                    iteration_index, []
                )
            )
            if not modules_data_for_iteration:
                continue

            iteration_visualizations: list[ModulVisualizationDto] = []

            for module_data in modules_data_for_iteration:
                if module_data.result is None:
                    continue

                visualization_content = self._render_module_visualization(
                    module_class=module_data.class_ref,
                    result=module_data.result,
                    task_execution_context=task_execution_context,
                    out_dir_path=path,
                )

                module_visualization = ModulVisualizationDto(
                    modul=module_data.class_ref,
                    package_type=module_data.package_type,
                    visualization_content=visualization_content,
                )

                if module_data.package_type == PackageType.Stat:
                    stats_visualizations.append(module_visualization)
                else:
                    iteration_visualizations.append(module_visualization)

            if iteration_visualizations:
                modules_visualizations_by_iteration[iteration_index] = (
                    iteration_visualizations
                )

        report_generator = ReportGenerator(task_execution_context)
        report_generator.export_task_report(
            output_format=self.output_format,
            modules_visualizations_by_iteration=modules_visualizations_by_iteration,
            stats_visualizations=stats_visualizations,
            out_dir_path=path,
        )

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

    @classmethod
    def get_short_name(cls) -> str:
        return "stat.report_task"

    @classmethod
    def get_long_name(cls) -> str:
        return "Task Report"

    @classmethod
    def get_description(cls) -> str:
        return "Generates a report for the given task."

    @classmethod
    def get_tags(cls) -> dict:
        return {"input": set(), "output": {"files"}}

    @classmethod
    def get_order(cls) -> int:
        return 100

    ####################################################################
    #########  Private functions
    ####################################################################
