from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, Template, select_autoescape

from fopimt.evaluators.evaluator import EvaluatorResult
from fopimt.task_dto import TaskExecutionContext

from .playthrough_2048_to_video import PlaythroughRenderer, VideoPayload


class Renderer2048:
    def __init__(self) -> None:
        self._rendering_root_dir_path = Path(__file__).resolve().parent

    def _get_template(self, template_name: str) -> Template:
        template_path = self._rendering_root_dir_path / "templates" / template_name
        if not template_path.exists():
            raise FileNotFoundError(
                f"Report template not found: {template_path.as_posix()}"
            )

        environment = Environment(
            loader=FileSystemLoader(str(template_path.parent)),
            autoescape=select_autoescape(["html", "xml"]),
        )
        return environment.get_template(template_path.name)

    def _get_result_iteration_index(
        self,
        evaluation_result: EvaluatorResult,
        task_execution_context: TaskExecutionContext,
    ) -> int | None:
        result_solution_id = evaluation_result.metadata.get("solution_id")
        if result_solution_id is None:
            logging.warning(
                "EvaluatorResult metadata does not contain 'solution_id'. Iteration index will be set to None."
            )
            return None
        for index, solution in enumerate(task_execution_context.solutions):
            if solution.get_id() == result_solution_id:
                return index

    def render_custom_visualization_html(
        self,
        evaluation_result: EvaluatorResult,
        task_execution_context: TaskExecutionContext,
        out_dir_path: str,
    ) -> str:
        logging.info(f"VISUALIZER ID | _render_custom_visualization: {id(self)}")

        metadata = evaluation_result.metadata
        if "results" not in metadata:
            logging.warning(
                "Visualizer2048: No 'results' found in solution metadata. Using empty results."
            )
            template = self._get_template("template_visualizer_2048_error.jinja2")
            return template.render(
                error_title="Visualization Error",
                error_message="No 'results' found in solution metadata.",
                error_details="The visualizer expected a 'results' field in the solution metadata, but it was not found. Ensure that the evaluator produces the expected output.",
                hint="Check logs and ensure the evaluator produced valid output artifacts.",
            )

        result_iteration_index = (
            self._get_result_iteration_index(evaluation_result, task_execution_context)
            or 0
        )

        videos_data = self._render_playthroughs(
            metadata.get("playthroughs", []),
            result_iteration_index,
            Path(out_dir_path),
        )

        visualizer_context = {
            "results": metadata.get("results", {}),
            "videos": videos_data,
        }

        template = self._get_template("template_visualizer_2048.jinja2")
        return template.render(**visualizer_context)

    def _render_playthroughs(
        self,
        playthroughs_data: list[list[dict[str, Any]]],
        current_solution_iteration: int,
        output_dir: Path,
    ) -> list[dict[str, str]]:
        logging.info("Visualizer2048: Rendering playthroughs...")

        if not playthroughs_data:
            logging.warning(
                "Visualizer2048: No playthrough data found in solution metadata."
            )
            return []

        # Render playthrough videos to in-memory payloads.
        videos_data = []
        renderer = PlaythroughRenderer(tile_size=110, font_size=32)
        for idx, playthrough_data in enumerate(playthroughs_data):
            video_payload: VideoPayload = renderer.generate_video_payload(
                playthrough_data,
                fps=30,
                seconds_per_state=0.05,
                macro_block_size=1,
            )
            video_output_path = (
                output_dir
                / "videos"
                / f"sol_{current_solution_iteration}_playthrough_{idx}.mp4"
            )
            PlaythroughRenderer.write_video(video_payload, video_output_path)
            videos_data.append(
                {
                    "src": f"videos/sol_{current_solution_iteration}_playthrough_{idx}.mp4",
                    "type": "video/mp4",
                    "title": f"Run {idx + 1}",
                }
            )

        logging.info("Visualizer2048: Rendering completed.")
        return videos_data
