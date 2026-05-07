import numpy as np
from .trajectories import Trajectory, TrajectoryFactory, Vec3

class PaintshopFunction:

    def __init__(self, funcNum: int):
        self._trajectory: Trajectory = TrajectoryFactory[funcNum]

    def evaluate(self, speed: list[float], distance: list[float]) -> float:
        params = {
            "w_dist": 1.0,
            "w_angle": 0.0,
            "w_speed": 1.0,
            "w_time": 1.0,
        }
        traj = self._trajectory
        for i in range(len(speed)):
            traj.points[i].speed = speed[i]
            traj.points[i].tcp_point = Vec3(traj.points[i].srf_point.x + (-distance[i]) * traj.points[i].direction.x,
                                            traj.points[i].srf_point.y + (-distance[i]) * traj.points[i].direction.y,
                                            traj.points[i].srf_point.z + (-distance[i]) * traj.points[i].direction.z)
        cost = self._calculate_inner_cost(traj, params)
        return cost

    def CF_extract(self) -> (list[float], list[float]):
        traj = self._trajectory
        x = list()
        y = list()
        for i in range(len(traj.points)):
            x.append(traj.points[i].speed)
            if traj.points[i].direction.x < 1e-6:
                y.append(0)
            else:
                y.append(-(traj.points[i].tcp_point.x - traj.points[i].srf_point.x) / traj.points[i].direction.x)
        return x, y

    # Returns number of dimensions
    def CF_info(self) -> int:
        return len(self._trajectory.points)

    def _calculate_inner_cost(self, trajectory: Trajectory, params: dict) -> float:
        """
        Compute the weighted cost for a single trajectory.

        params keys:
            w_dist   - weight for distance-to-surface penalty
            w_angle  - weight for angle penalty (placeholder, always 0)
            w_speed  - weight for speed penalty
            w_time   - weight for total travel time
        """

        IDEAL_DIST = 250.0  # mm
        IDEAL_SPEED = 500.0  # mm/s

        total_cost = 0.0
        total_time = 0.0

        for i, pt in enumerate(trajectory.points):
            tcp = np.array([pt.tcp_point.x, pt.tcp_point.y, pt.tcp_point.z])
            srf = np.array([pt.srf_point.x, pt.srf_point.y, pt.srf_point.z])

            # 1. Distance penalty
            dist = np.linalg.norm(tcp - srf)
            dist_penalty = (dist - IDEAL_DIST) ** 2

            # 2. Angle penalty (placeholder — no surface normal available yet)
            angle_penalty = 0.0

            # 3. Speed penalty
            speed_penalty = (pt.speed - IDEAL_SPEED) ** 2

            # 4. Time between consecutive waypoints
            if i > 0:
                prev = trajectory.points[i - 1]
                prev_tcp = np.array([prev.tcp_point.x, prev.tcp_point.y, prev.tcp_point.z])
                step_dist = np.linalg.norm(tcp - prev_tcp)
                total_time += step_dist / (pt.speed + 1e-6)

            total_cost += (
                    params["w_dist"] * dist_penalty
                    + params["w_angle"] * angle_penalty
                    + params["w_speed"] * speed_penalty
            )

        total_cost += params["w_time"] * total_time

        return total_cost
