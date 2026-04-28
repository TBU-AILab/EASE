from __future__ import annotations

import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, cast

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

Move = str


@dataclass(frozen=True)
class PlayState:
    board: list[list[int]]
    score: int
    move: Optional[Move] = None


@dataclass(frozen=True)
class VideoPayload:
    frames: list[np.ndarray]
    fps: int
    macro_block_size: int = 1
    codec: Optional[str] = None


class PlaythroughRenderer:
    TILE_COLORS: dict[int, str] = {
        0: "#CDC1B4",
        2: "#EEE4DA",
        4: "#EDE0C8",
        8: "#F2B179",
        16: "#F59563",
        32: "#F67C5F",
        64: "#F65E3B",
        128: "#EDCF72",
        256: "#EDCC61",
        512: "#EDC850",
        1024: "#EDC53F",
        2048: "#EDC22E",
    }
    TEXT_COLORS: dict[int, str] = {
        2: "#776E65",
        4: "#776E65",
        8: "#F9F6F2",
        16: "#F9F6F2",
        32: "#F9F6F2",
        64: "#F9F6F2",
        128: "#F9F6F2",
        256: "#F9F6F2",
        512: "#F9F6F2",
        1024: "#F9F6F2",
        2048: "#F9F6F2",
    }

    def __init__(
        self,
        *,
        grid_size: int = 4,
        tile_size: int = 100,
        margin: int = 10,
        footer_height: int = 56,
        font_size: int = 56,
        background_color: str = "#BBADA0",
        empty_tile_color: str = "#CDC1B4",
        high_tile_color: str = "#3C3A32",
        score_text_color: str = "#776E65",
        move_text_color: str = "#776E65",
    ) -> None:
        self.grid_size = self._validate_positive_int(grid_size, "grid_size")
        self.tile_size = self._validate_positive_int(tile_size, "tile_size")
        self.margin = self._validate_positive_int(margin, "margin")
        self.footer_height = self._validate_positive_int(footer_height, "footer_height")

        self.background_color = background_color
        self.empty_tile_color = empty_tile_color
        self.high_tile_color = high_tile_color
        self.score_text_color = score_text_color
        self.move_text_color = move_text_color

        self.width = (
            self.grid_size * self.tile_size + (self.grid_size + 1) * self.margin
        )
        self.height = self.width + self.footer_height

        font_path = Path(__file__).resolve().parent / "assets" / "Inter-Medium.ttf"
        if not font_path.exists():
            raise FileNotFoundError(f"Required font not found: {font_path}\n")

        self.font = ImageFont.truetype(
            str(font_path), size=self._validate_positive_int(font_size, "font_size")
        )

    def generate_video_payload(
        self,
        states: Sequence[PlayState] | Sequence[Mapping[str, Any]],
        *,
        fps: int = 30,
        seconds_per_state: float = 0.05,
        macro_block_size: int = 1,
        codec: Optional[str] = None,
    ) -> VideoPayload:
        fps = self._validate_positive_int(fps, "fps")
        if seconds_per_state <= 0:
            raise ValueError("seconds_per_state must be > 0.")
        if macro_block_size <= 0:
            raise ValueError("macro_block_size must be > 0.")

        typed_states = self._coerce_states(states)
        self._validate_states(typed_states)

        repeats = max(1, int(round(fps * seconds_per_state)))
        frames: list[np.ndarray] = []

        for st in typed_states:
            img = self._render_state(st)
            frame = np.asarray(img, dtype=np.uint8)
            for _ in range(repeats):
                frames.append(frame)

        return VideoPayload(
            frames=frames,
            fps=fps,
            macro_block_size=macro_block_size,
            codec=codec,
        )

    @staticmethod
    def write_video(payload: VideoPayload, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        writer_kwargs: dict[str, Any] = {
            "fps": payload.fps,
            "macro_block_size": payload.macro_block_size,
        }
        if payload.codec:
            writer_kwargs["codec"] = payload.codec

        with imageio.get_writer(str(output_path), **writer_kwargs) as writer:
            writer_obj: Any = writer
            for frame in payload.frames:
                writer_obj.append_data(frame)

    def _render_state(self, state: PlayState) -> Image.Image:
        img = Image.new("RGB", (self.width, self.height), self.background_color)
        draw = ImageDraw.Draw(img)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                val = state.board[i][j]

                tile_color = (
                    self.empty_tile_color
                    if val == 0
                    else self.TILE_COLORS.get(val, self.high_tile_color)
                )
                text_color = self.TEXT_COLORS.get(val, "#F9F6F2")

                x0 = j * self.tile_size + (j + 1) * self.margin
                y0 = i * self.tile_size + (i + 1) * self.margin
                x1 = x0 + self.tile_size
                y1 = y0 + self.tile_size

                draw.rectangle(
                    [x0, y0, x1, y1], fill=tile_color, outline=self.background_color
                )

                if val:
                    self._draw_centered_text(
                        draw=draw,
                        box=(x0, y0, x1, y1),
                        text=str(val),
                        fill=text_color,
                        font=self.font,
                    )

        footer_center_y = self.height - self.footer_height / 2

        self._draw_text_left_centered(
            draw=draw,
            pos=(self.margin, footer_center_y),
            text=f"Score: {state.score}",
            fill=self.score_text_color,
            font=self.font,
        )

        if state.move:
            move_text = f"Move: {state.move}"
            move_w, _ = self._text_size(draw, move_text, self.font)
            x = self.width - self.margin - move_w
            self._draw_text_left_centered(
                draw=draw,
                pos=(x, footer_center_y),
                text=move_text,
                fill=self.move_text_color,
                font=self.font,
            )

        return img

    @staticmethod
    def _draw_centered_text(
        *,
        draw: ImageDraw.ImageDraw,
        box: tuple[float, float, float, float],
        text: str,
        fill: str,
        font: Any,
    ) -> None:
        x0, y0, x1, y1 = box
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = x0 + ((x1 - x0) - tw) / 2
        y = y0 + ((y1 - y0) - th) / 2
        draw.text((x, y), text, fill=fill, font=font)

    @staticmethod
    def _draw_text_left_centered(
        *,
        draw: ImageDraw.ImageDraw,
        pos: tuple[float, float],
        text: str,
        fill: str,
        font: Any,
    ) -> None:
        x, center_y = pos
        bbox = draw.textbbox((0, 0), text, font=font)
        th = bbox[3] - bbox[1]
        y = center_y - th / 2
        draw.text((x, y), text, fill=fill, font=font)

    @staticmethod
    def _text_size(draw: ImageDraw.ImageDraw, text: str, font: Any) -> tuple[int, int]:
        bbox = draw.textbbox((0, 0), text, font=font)
        return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])

    def _coerce_states(
        self,
        states: Sequence[PlayState] | Sequence[Mapping[str, Any]],
    ) -> list[PlayState]:
        if not states:
            return []

        first = states[0]
        if isinstance(first, PlayState):
            return list(cast(Sequence[PlayState], states))

        out: list[PlayState] = []
        for idx, item in enumerate(states):
            if not isinstance(item, Mapping):
                raise ValueError(f"State #{idx} must be a PlayState or a mapping/dict.")

            try:
                board = item["board"]
                score = item["score"]
                move = item.get("move")
            except KeyError as e:
                raise ValueError(f"State #{idx} missing required key: {e}") from e

            if not isinstance(score, numbers.Integral):
                raise ValueError(
                    f"State #{idx} score must be int, got {type(score).__name__}."
                )
            if move is not None and not isinstance(move, str):
                raise ValueError(
                    f"State #{idx} move must be str or null, got {type(move).__name__}."
                )

            out.append(
                PlayState(
                    board=self._coerce_board(board, idx), score=int(score), move=move
                )
            )

        return out

    def _coerce_board(self, board: Any, idx: int) -> list[list[int]]:
        if not isinstance(board, list) or len(board) != self.grid_size:
            raise ValueError(
                f"State #{idx} board must be a list of {self.grid_size} rows."
            )

        out: list[list[int]] = []
        for r, row in enumerate(board):
            if not isinstance(row, list) or len(row) != self.grid_size:
                raise ValueError(
                    f"State #{idx} board row {r} must be a list of {self.grid_size} ints."
                )

            out_row: list[int] = []
            for c, cell in enumerate(row):
                if not isinstance(cell, int):
                    raise ValueError(
                        f"State #{idx} board[{r}][{c}] must be int, got {type(cell).__name__}."
                    )
                if cell < 0:
                    raise ValueError(f"State #{idx} board[{r}][{c}] must be >= 0.")
                out_row.append(cell)

            out.append(out_row)

        return out

    def _validate_states(self, states: Sequence[PlayState]) -> None:
        if not states:
            raise ValueError("No states provided (empty playthrough).")

        for idx, st in enumerate(states):
            if len(st.board) != self.grid_size:
                raise ValueError(f"State #{idx} board has wrong number of rows.")
            for r, row in enumerate(st.board):
                if len(row) != self.grid_size:
                    raise ValueError(
                        f"State #{idx} board row {r} has wrong number of columns."
                    )

    @staticmethod
    def _validate_positive_int(value: int, name: str) -> int:
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
        return value
