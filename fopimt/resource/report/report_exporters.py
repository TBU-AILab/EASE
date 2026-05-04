import shutil
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import ClassVar

from jinja2 import (
    ChoiceLoader,
    Environment,
    FileSystemLoader,
    Template,
    select_autoescape,
)

from fopimt.utils.render_utils import latex_escape

from .report_data_builders import (
    SolutionReportData,
    TaskReportData,
)


class BaseReportExporter(ABC):
    OUTPUT_FILENAME: ClassVar[str]

    def __init__(self, resource_root_dir_path: Path) -> None:
        self._resource_root_dir_path = resource_root_dir_path

    @abstractmethod
    def export(
        self,
        report_data: SolutionReportData | TaskReportData,
        out_dir_path: str,
        template_name: str,
    ) -> None:
        pass

    @abstractmethod
    def _get_template(self, template_path: Path) -> Template:
        pass

    def _render_to_file(
        self,
        report_data: SolutionReportData | TaskReportData,
        out_dir_path: str,
        template_path: Path,
    ) -> None:
        template = self._get_template(template_path)
        out_file_path = Path(out_dir_path) / self.OUTPUT_FILENAME
        out_file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_file_path, "w", encoding="utf-8") as f:
            f.write(template.render(**asdict(report_data)))


class HtmlReportExporter(BaseReportExporter):
    OUTPUT_FILENAME: ClassVar[str] = "report.html"
    _USED_ASSETS: ClassVar[list[str]] = ["plotly.min.js"]

    def export(
        self,
        report_data: SolutionReportData | TaskReportData,
        out_dir_path: str,
        template_name: str,
    ) -> None:
        self._export_assets(out_dir_path)
        template_path = (
            self._resource_root_dir_path / "templates" / "html" / template_name
        )
        self._render_to_file(report_data, out_dir_path, template_path)

    def _get_template(self, template_path: Path) -> Template:
        if not template_path.exists():
            raise FileNotFoundError(
                f"Report template not found: {template_path.as_posix()}"
            )

        core_shared_dir = (
            self._resource_root_dir_path.parents[1] / "templates" / "html" / "shared"
        )

        environment = Environment(
            loader=ChoiceLoader(
                [
                    FileSystemLoader(str(template_path.parent)),
                    FileSystemLoader(str(core_shared_dir)),
                ]
            ),
            autoescape=select_autoescape(["html", "xml"]),
        )

        return environment.get_template(template_path.name)

    def _export_assets(self, out_dir_path: str) -> None:
        assets_dir = self._resource_root_dir_path / "assets"
        out_assets_dir = Path(out_dir_path) / "assets"
        out_assets_dir.mkdir(parents=True, exist_ok=True)

        for asset in self._USED_ASSETS:
            asset_path = assets_dir / asset
            if not asset_path.exists():
                raise FileNotFoundError(
                    f"Report asset not found: {asset_path.as_posix()}"
                )
            shutil.copy2(asset_path, out_assets_dir / asset)


class LatexReportExporter(BaseReportExporter):
    OUTPUT_FILENAME: ClassVar[str] = "report.tex"

    def export(
        self,
        report_data: SolutionReportData | TaskReportData,
        out_dir_path: str,
        template_name: str,
    ) -> None:
        template_path = (
            self._resource_root_dir_path / "templates" / "latex" / template_name
        )
        self._render_to_file(report_data, out_dir_path, template_path)

    def _get_template(self, template_path: Path) -> Template:
        if not template_path.exists():
            raise FileNotFoundError(
                f"Report template not found: {template_path.as_posix()}"
            )

        environment = Environment(
            loader=FileSystemLoader(str(template_path.parent)),
            autoescape=False,
        )
        environment.filters["latex_escape"] = latex_escape

        return environment.get_template(template_path.name)
