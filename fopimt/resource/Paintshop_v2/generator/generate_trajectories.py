"""
Generator script - run once to produce trajectories.py with hardcoded data.
Usage: python generate_trajectories.py
"""
import json
import os

FILES_ORDER = [
    "MAIN_TRAJECTORY.json",
    "P1_TRAJECTORY.json",
    "P2_TRAJECTORY.json",
    "P3_TRAJECTORY.json",
    "P4_TRAJECTORY.json",
    "P5_TRAJECTORY.json",
]

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def v3(d):
    return f"Vec3({d['x']}, {d['y']}, {d['z']})"


def build_set_brush(sb):
    info = sb.get("Info")
    if info:
        return (
            f"SetBrush({sb['RecordNumber']}, {sb['Type']}, {sb['Position']}, "
            f"{info['SBWidth']}, {info['MaxThickness']}, {info['TransferEfficiency']})"
        )
    return f"SetBrush({sb['RecordNumber']}, {sb['Type']}, {sb['Position']})"


def build_point(pt):
    tcp = v3(pt["TCPPoint"])
    srf = v3(pt["SRFPoint"])
    direction = v3(pt["Direction"])
    speed = pt["Speed"]
    brushes = pt.get("SetBrushes", [])
    if brushes:
        brush_list = "[" + ", ".join(build_set_brush(b) for b in brushes) + "]"
        return f"        TrajectoryPoint({speed}, {tcp}, {srf}, {direction}, {brush_list}),"
    return f"        TrajectoryPoint({speed}, {tcp}, {srf}, {direction}),"


def main():
    all_trajectories = []

    for filename in FILES_ORDER:
        path = os.path.join(DATA_DIR, filename)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for traj in data.get("Trajectories", []):
            points = [build_point(p) for p in traj["Points"]]
            all_trajectories.append(points)

    lines = [
        "# AUTO-GENERATED — do not edit by hand.",
        "# Re-run generate_trajectories.py to regenerate.",
        "from __future__ import annotations",
        "from dataclasses import dataclass, field",
        "from typing import Optional, List",
        "",
        "",
        "@dataclass",
        "class Vec3:",
        "    x: float",
        "    y: float",
        "    z: float",
        "",
        "",
        "@dataclass",
        "class SetBrush:",
        "    record_number: int",
        "    type: int",
        "    position: float",
        "    sb_width: Optional[float] = None",
        "    max_thickness: Optional[float] = None",
        "    transfer_efficiency: Optional[float] = None",
        "",
        "",
        "@dataclass",
        "class TrajectoryPoint:",
        "    speed: float",
        "    tcp_point: Vec3",
        "    srf_point: Vec3",
        "    direction: Vec3",
        "    set_brushes: List[SetBrush] = field(default_factory=list)",
        "",
        "",
        "@dataclass",
        "class Trajectory:",
        "    points: List[TrajectoryPoint]",
        "",
        "",
        "class TrajectoryFactory:",
        "    _trajectories: List[Trajectory] = [",
    ]

    for traj_points in all_trajectories:
        lines.append("        Trajectory([")
        lines.extend(traj_points)
        lines.append("        ]),")

    lines += [
        "    ]",
        "",
        "    @classmethod",
        "    def count(cls) -> int:",
        '        """Return the number of available trajectories."""',
        "        return len(cls._trajectories)",
        "",
        "    @classmethod",
        "    def get(cls, index: int) -> Trajectory:",
        '        """Return the trajectory at the given index (0-based)."""',
        "        return cls._trajectories[index]",
        "",
        "    def __class_getitem__(cls, index: int) -> Trajectory:",
        '        """Allow TrajectoryFactory[i] syntax."""',
        "        return cls._trajectories[index]",
        "",
    ]

    out_path = os.path.join(os.path.dirname(__file__), "trajectories.py")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Generated {out_path}")
    print(f"  Total trajectories : {len(all_trajectories)}")
    for i, pts in enumerate(all_trajectories):
        print(f"  Trajectory {i:2d}      : {len(pts)} points")


if __name__ == "__main__":
    main()
