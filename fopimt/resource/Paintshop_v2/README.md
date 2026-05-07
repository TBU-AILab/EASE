# Paintshop Cost

A Python toolset for evaluating robot paint-gun trajectories against a weighted cost function.

---

## Repository layout

```
data/
    MAIN_TRAJECTORY.json   # full-car trajectory (884 points)
    P1_TRAJECTORY.json     # panel 1 trajectory
    P2_TRAJECTORY.json     # panel 2 trajectory
    P3_TRAJECTORY.json     # panel 3 trajectory
    P4_TRAJECTORY.json     # panel 4 (3 sub-trajectories)
    P5_TRAJECTORY.json     # panel 5 trajectory
    *.BRUSH.json           # brush-configuration files (not used here)

generate_trajectories.py   # generator – converts JSON data → trajectories.py
trajectories.py            # AUTO-GENERATED – hardcoded trajectory data + factory
cost.py                    # cost function + evaluation main
```

---

## Data model (`trajectories.py`)

The file is **auto-generated** by `generate_trajectories.py` and must not be edited by hand.

| Class | Fields |
|---|---|
| `Vec3` | `x`, `y`, `z : float` |
| `SetBrush` | `record_number`, `type`, `position`, optional `sb_width`, `max_thickness`, `transfer_efficiency` |
| `TrajectoryPoint` | `speed`, `tcp_point : Vec3`, `srf_point : Vec3`, `direction : Vec3`, `set_brushes : List[SetBrush]` |
| `Trajectory` | `points : List[TrajectoryPoint]` |

Each `TrajectoryPoint` maps directly to one robot waypoint:

- **`tcp_point`** – gun (Tool Centre Point) position in mm
- **`srf_point`** – target surface point in mm
- **`direction`** – unit vector for gun orientation
- **`speed`** – TCP speed in mm/s
- **`set_brushes`** – brush-on / brush-off events at this point (may be empty)

### TrajectoryFactory

```python
TrajectoryFactory.count()     # → int  (number of trajectories)
TrajectoryFactory.get(i)      # → Trajectory
TrajectoryFactory[i]          # → Trajectory  (indexer shorthand)
```

Trajectories are ordered as follows:

| Index | Source file | Points |
|---|---|---|
| 0 | MAIN_TRAJECTORY.json | 884 |
| 1 | P1_TRAJECTORY.json | 177 |
| 2 | P2_TRAJECTORY.json | 171 |
| 3 | P3_TRAJECTORY.json | 181 |
| 4 | P4_TRAJECTORY.json \[0\] | 58 |
| 5 | P4_TRAJECTORY.json \[1\] | 58 |
| 6 | P4_TRAJECTORY.json \[2\] | 58 |
| 7 | P5_TRAJECTORY.json | 182 |

---

## Cost function (`cost.py`)

```python
calculate_inner_cost(trajectory: Trajectory, params: dict) -> float
```

Evaluates a weighted sum of penalties over every waypoint in the trajectory.

### Terms

| Term | Formula | Ideal constant |
|---|---|---|
| Distance penalty | `(‖tcp − srf‖ − IDEAL_DIST)²` | `IDEAL_DIST = 250 mm` |
| Angle penalty | placeholder – always 0 | – |
| Speed penalty | `(speed − IDEAL_SPEED)²` | `IDEAL_SPEED = 500 mm/s` |
| Travel time | `Σ ‖tcp_i − tcp_{i-1}‖ / speed_i` | – |

### Total cost

```
cost = Σ_i ( w_dist · dist_penalty_i
           + w_angle · angle_penalty_i
           + w_speed · speed_penalty_i )
     + w_time · total_time
```

### `params` dictionary

```python
params = {
    "w_dist":  1.0,   # weight for distance penalty
    "w_angle": 0.0,   # weight for angle penalty (not yet implemented)
    "w_speed": 1.0,   # weight for speed penalty
    "w_time":  1.0,   # weight for travel-time term
}
```

---

## Regenerating trajectory data

If the source JSON files change, re-run the generator:

```bash
python generate_trajectories.py
```

This overwrites `trajectories.py` with freshly hardcoded data.

---

## Running the evaluation

```bash
python cost.py
```

Prints a table of cost values for all 8 trajectories using the default weights.

```
  Trajectory  Points                Cost
------------------------------------------
           0     884        8254648.60
           1     177       17652536.20
           ...
```

---

## Dependencies

- Python 3.9+
- `numpy`
