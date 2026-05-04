from dataclasses import dataclass
from enum import Enum

from fopimt.loader_dto import PackageType
from fopimt.modul import Modul


@dataclass
class ModulVisualizationDto:
    modul: type[Modul]
    package_type: PackageType
    visualization_content: str | None = None


class OutputFormat(str, Enum):
    HTML = "html"
    LATEX = "latex"

    @property
    def render_method_name(self) -> str:
        render_method_by_format: dict["OutputFormat", str] = {
            OutputFormat.HTML: "render_html",
            OutputFormat.LATEX: "render_latex",
        }
        return render_method_by_format[self]
