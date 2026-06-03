from dataclasses import asdict, dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Aperture:
    """Integer pixel window on the full simulation canvas.

    The slice convention is the same as PyTorch/Numpy: y0:y1 and x0:x1,
    where the end index is excluded.
    """

    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def height(self) -> int:
        return int(self.y1 - self.y0)

    @property
    def width(self) -> int:
        return int(self.x1 - self.x0)

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.y0 + self.y1) / 2.0, (self.x0 + self.x1) / 2.0)

    def shifted(self, dy: int = 0, dx: int = 0) -> "Aperture":
        return Aperture(self.y0 + int(dy), self.y1 + int(dy), self.x0 + int(dx), self.x1 + int(dx))

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class MoeLayout:
    """Large-canvas layout for two side-by-side optical experts."""

    canvas_height: int = 800
    canvas_width: int = 1600
    expert_size: int = 600
    gap_pixels: int = 200
    margin_x: int = 100
    margin_y: int = 100
    input_size: int = 200

    @property
    def canvas_shape(self) -> Tuple[int, int]:
        return (int(self.canvas_height), int(self.canvas_width))

    @property
    def center_y(self) -> float:
        return self.canvas_height / 2.0

    @property
    def center_x(self) -> float:
        return self.canvas_width / 2.0

    @property
    def center(self) -> Tuple[float, float]:
        return (self.center_y, self.center_x)

    @property
    def left(self) -> Aperture:
        return Aperture(
            self.margin_y,
            self.margin_y + self.expert_size,
            self.margin_x,
            self.margin_x + self.expert_size,
        )

    @property
    def right(self) -> Aperture:
        x0 = self.margin_x + self.expert_size + self.gap_pixels
        return Aperture(self.margin_y, self.margin_y + self.expert_size, x0, x0 + self.expert_size)

    @property
    def left_shift_pixels(self) -> float:
        return self.left.center[1] - self.center_x

    @property
    def right_shift_pixels(self) -> float:
        return self.right.center[1] - self.center_x

    @property
    def input_aperture(self) -> Aperture:
        y0 = int(round(self.center_y - self.input_size / 2.0))
        x0 = int(round(self.center_x - self.input_size / 2.0))
        return Aperture(y0, y0 + self.input_size, x0, x0 + self.input_size)

    def aperture_for_side(self, side: str) -> Aperture:
        if side == "left":
            return self.left
        if side == "right":
            return self.right
        raise ValueError("side must be 'left' or 'right'")

    def target_center(self, side: str) -> Tuple[float, float]:
        return self.aperture_for_side(side).center

    def validate(self) -> None:
        if self.left.height != self.expert_size or self.left.width != self.expert_size:
            raise ValueError("Left aperture does not match expert_size.")
        if self.right.height != self.expert_size or self.right.width != self.expert_size:
            raise ValueError("Right aperture does not match expert_size.")
        if self.left.x1 > self.right.x0:
            raise ValueError("Left and right expert apertures overlap.")
        for name, aperture in [("left", self.left), ("right", self.right), ("input", self.input_aperture)]:
            if aperture.y0 < 0 or aperture.x0 < 0:
                raise ValueError(f"{name} aperture starts outside the canvas.")
            if aperture.y1 > self.canvas_height or aperture.x1 > self.canvas_width:
                raise ValueError(f"{name} aperture ends outside the canvas.")

    def to_dict(self) -> Dict:
        return {
            "canvas_shape": list(self.canvas_shape),
            "canvas_center": [self.center_y, self.center_x],
            "expert_size": self.expert_size,
            "gap_pixels": self.gap_pixels,
            "margin_x": self.margin_x,
            "margin_y": self.margin_y,
            "input_size": self.input_size,
            "input_aperture": self.input_aperture.to_dict(),
            "left_aperture": self.left.to_dict(),
            "right_aperture": self.right.to_dict(),
            "left_shift_pixels": self.left_shift_pixels,
            "right_shift_pixels": self.right_shift_pixels,
        }


def build_moe_layout(config: Dict) -> MoeLayout:
    """Build a layout from a YAML config section while keeping defaults explicit."""

    layout = MoeLayout(
        canvas_height=int(config.get("canvas_height", 800)),
        canvas_width=int(config.get("canvas_width", 1600)),
        expert_size=int(config.get("expert_size", 600)),
        gap_pixels=int(config.get("gap_pixels", 200)),
        margin_x=int(config.get("margin_x", 100)),
        margin_y=int(config.get("margin_y", 100)),
        input_size=int(config.get("input_size", 200)),
    )
    layout.validate()
    return layout
