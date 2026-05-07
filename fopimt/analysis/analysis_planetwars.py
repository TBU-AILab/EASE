import json
import os
import shutil
from pathlib import Path
from typing import Any

import requests

from ..loader_dto import Parameter, PrimitiveType
from ..solutions.solution import Solution
from ..task_dto import TaskExecutionContext
from .analysis import Analysis, AnalysisResult


class AnalysisPlanetWarsVideos(Analysis):
    """
    Generates N MP4 videos of a PlanetWars solution playing against a baseline agent.
    The actual execution and video generation happen in the planetwars-evaluator service.
    """

    @classmethod
    def get_parameters(cls) -> dict[str, Parameter]:
        return {
            "service_url": Parameter(
                short_name="service_url",
                type=PrimitiveType.str,
                long_name="PlanetWars evaluator service URL",
                description="Inside Docker Compose use http://planetwars-evaluator:8090.",
                default="http://planetwars-evaluator:8090",
            ),
            "class_name": Parameter(
                short_name="class_name",
                type=PrimitiveType.str,
                long_name="Agent class name",
                description="Name of the generated candidate class.",
                default="MyAgent",
            ),
            "n_videos": Parameter(
                short_name="n_videos",
                type=PrimitiveType.int,
                long_name="Number of videos",
                description="Number of games/videos to generate.",
                default=3,
                min_value=1,
                max_value=50,
            ),
            "opponent": Parameter(
                short_name="opponent",
                type=PrimitiveType.enum,
                long_name="Opponent",
                description="Baseline opponent for video generation.",
                enum_options=[
                    "PureRandomAgent",
                    "CarefulRandomAgent",
                    "GreedyHeuristicAgent",
                ],
                default="GreedyHeuristicAgent",
            ),
            "candidate_player": Parameter(
                short_name="candidate_player",
                type=PrimitiveType.enum,
                long_name="Candidate player side",
                description="Side used by the generated agent in the videos.",
                enum_options=["Player1", "Player2"],
                default="Player1",
            ),
            "partial_observability": Parameter(
                short_name="partial_observability",
                type=PrimitiveType.bool,
                long_name="Partial observability",
                description="Whether to run videos in partial-observation mode.",
                default=False,
            ),
            "num_planets": Parameter(
                short_name="num_planets",
                type=PrimitiveType.int,
                long_name="Number of planets",
                description="Planet Wars map size.",
                default=20,
                min_value=2,
                max_value=200,
            ),
            "max_ticks": Parameter(
                short_name="max_ticks",
                type=PrimitiveType.int,
                long_name="Maximum ticks",
                description="Maximum game duration.",
                default=2000,
                min_value=1,
                max_value=100000,
            ),
            "fps": Parameter(
                short_name="fps",
                type=PrimitiveType.int,
                long_name="Video FPS",
                description="Frames per second in the generated MP4 videos.",
                default=20,
                min_value=1,
                max_value=60,
            ),
            "frame_stride": Parameter(
                short_name="frame_stride",
                type=PrimitiveType.int,
                long_name="Frame stride",
                description="Capture one frame every N game ticks.",
                default=2,
                min_value=1,
                max_value=100,
            ),
            "seed": Parameter(
                short_name="seed",
                type=PrimitiveType.int,
                long_name="Seed",
                description="Base seed for rendered games.",
                default=12345,
            ),
            "timeout_sec": Parameter(
                short_name="timeout_sec",
                type=PrimitiveType.int,
                long_name="Render timeout",
                description="Timeout for the render service.",
                default=300,
                min_value=1,
                max_value=3600,
            ),
            "request_timeout_sec": Parameter(
                short_name="request_timeout_sec",
                type=PrimitiveType.int,
                long_name="HTTP timeout",
                description="Timeout for the HTTP request from EASE to the render service.",
                default=360,
                min_value=1,
                max_value=3600,
            ),
            "reports_dir": Parameter(
                short_name="reports_dir",
                type=PrimitiveType.str,
                long_name="Shared reports directory",
                description="Shared volume path visible from backend-core.",
                default="/planetwars_reports",
            ),
        }

    def _init_params(self):
        super()._init_params()

        self._service_url = self.parameters.get(
            "service_url",
            os.getenv("PLANETWARS_EVALUATOR_URL", "http://planetwars-evaluator:8090"),
        ).rstrip("/")

        self._class_name = self.parameters.get("class_name", "MyAgent")
        self._n_videos = int(self.parameters.get("n_videos", 3))
        self._opponent = self.parameters.get("opponent", "GreedyHeuristicAgent")
        self._candidate_player = self.parameters.get("candidate_player", "Player1")
        self._partial_observability = bool(
            self.parameters.get("partial_observability", False)
        )

        self._num_planets = int(self.parameters.get("num_planets", 20))
        self._max_ticks = int(self.parameters.get("max_ticks", 2000))

        self._fps = int(self.parameters.get("fps", 20))
        self._frame_stride = int(self.parameters.get("frame_stride", 2))
        self._seed = int(self.parameters.get("seed", 12345))
        self._timeout_sec = int(self.parameters.get("timeout_sec", 300))
        self._request_timeout_sec = int(
            self.parameters.get("request_timeout_sec", 360)
        )
        self._reports_dir = self.parameters.get(
            "reports_dir",
            os.getenv("PLANETWARS_REPORTS_DIR", "/planetwars_reports"),
        )

        self._result: dict[str, Any] | None = None

    def evaluate_analysis(
        self,
        solution: Solution,
        task_execution_context: TaskExecutionContext,
    ) -> AnalysisResult:
        agent_code = solution.get_input() or ""

        payload = {
            "agent_code": agent_code,
            "class_name": self._class_name,
            "n_videos": self._n_videos,
            "opponent": self._opponent,
            "candidate_player": self._candidate_player,
            "partial_observability": self._partial_observability,
            "seed": self._seed,
            "timeout_sec": self._timeout_sec,
            "fps": self._fps,
            "frame_stride": self._frame_stride,
            "game_params": {
                "num_planets": self._num_planets,
                "max_ticks": self._max_ticks,
                "new_map_each_run": True,
            },
        }

        try:
            response = requests.post(
                f"{self._service_url}/render_videos",
                json=payload,
                timeout=self._request_timeout_sec,
            )
            response.raise_for_status()
            self._result = response.json()
        except Exception as exc:
            self._result = {
                "ok": False,
                "errors": [
                    "Could not generate PlanetWars videos.",
                    repr(exc),
                ],
                "videos": [],
            }

        return AnalysisResult(
            class_ref=type(self),
            metadata=self._result,
        )

    def export(self, path: str, id: str) -> None:
        if self._result is None:
            raise RuntimeError("No PlanetWars video analysis result available.")

        export_root = Path(path) / f"{id}_{self.get_short_name()}"
        export_root.mkdir(parents=True, exist_ok=True)

        copied_videos: list[dict[str, Any]] = []

        for video in self._result.get("videos", []):
            src = Path(video["path"])
            dst = export_root / src.name

            if src.exists():
                shutil.copy2(src, dst)

                copied = dict(video)
                copied["exported_filename"] = dst.name
                copied_videos.append(copied)

        index_html = self._make_index_html(copied_videos, self._result)

        (export_root / "index.html").write_text(index_html, encoding="utf-8")
        (export_root / "planetwars_video_result.json").write_text(
            json.dumps(self._result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def get_feedback(self) -> str:
        if not self._result:
            return ""

        if not self._result.get("ok", False):
            return "PlanetWars video analysis failed."

        videos = self._result.get("videos", [])
        wins = sum(1 for item in videos if item.get("candidate_win"))
        return f"PlanetWars video analysis generated {len(videos)} videos, wins: {wins}."

    @classmethod
    def get_short_name(cls) -> str:
        return "anal.planetwars_videos"

    @classmethod
    def get_long_name(cls) -> str:
        return "PlanetWars videos"

    @classmethod
    def get_description(cls) -> str:
        return (
            "Generates MP4 videos of the PlanetWars solution playing against "
            "a selected baseline opponent."
        )

    @classmethod
    def get_tags(cls) -> dict:
        return {"input": {"python"}, "output": {"files"}}

    @staticmethod
    def _make_index_html(
        videos: list[dict[str, Any]],
        result: dict[str, Any],
    ) -> str:
        rows = []

        for idx, video in enumerate(videos, start=1):
            filename = video["exported_filename"]
            winner = video.get("winner", "?")
            ticks = video.get("ticks", "?")
            candidate_win = video.get("candidate_win", False)

            rows.append(
                f"""
                <section style="margin-bottom: 2rem;">
                    <h2>Game {idx}</h2>
                    <p>
                        Winner: <strong>{winner}</strong><br>
                        Candidate win: <strong>{candidate_win}</strong><br>
                        Ticks: <strong>{ticks}</strong>
                    </p>
                    <video width="720" controls>
                        <source src="{filename}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                </section>
                """
            )

        errors = result.get("errors", [])
        errors_html = ""
        if errors:
            errors_html = "<h2>Errors</h2><pre>" + "\n".join(map(str, errors)) + "</pre>"

        return f"""
        <!doctype html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>PlanetWars Video Report</title>
        </head>
        <body>
            <h1>PlanetWars Video Report</h1>
            <p>Generated videos: {len(videos)}</p>
            {errors_html}
            {''.join(rows)}
        </body>
        </html>
        """