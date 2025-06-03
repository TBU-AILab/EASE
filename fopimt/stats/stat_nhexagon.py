from .stat import Stat
from ..solutions.solution import Solution
import numpy as np
import pandas as pd
from scipy.stats import ranksums
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import sys

def hexagon_vertices(
    center_x: float,
    center_y: float,
    side_length: float,
    angle_degrees: float,
) -> list[tuple[float, float]]:
    """Generates the vertices of a regular hexagon.

    Args:
    center_x: x-coordinate of the center.
    center_y: y-coordinate of the center.
    side_length: Length of each side.
    angle_degrees: Rotation angle in degrees (clockwise from horizontal).

    Returns:
    A list of tuples, where each tuple (x, y) represents the vertex location.
    """
    vertices = []
    angle_radians = math.radians(angle_degrees)
    for i in range(6):
        angle = angle_radians + 2 * math.pi * i / 6
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices.append((x, y))
    return vertices

def plot_construction(
    inner_hex_data: list[tuple[float, float, float]],
    outer_hex_center: tuple[float, float],
    outer_hex_side_length: float,
    outer_hex_angle_degrees: float,
    filepath: str
):
  """Plots the hexagon packing construction using matplotlib.

  Args:
    inner_hex_data: List of (x, y, angle_degrees) tuples for inner hexagons.
    outer_hex_center: (x, y) tuple for the outer hexagon center.
    outer_hex_side_length: Side length of the outer hexagon.
    outer_hex_angle_degrees: Rotation angle of the outer hexagon in degrees.
  """

  inner_hex_params_list = [(x, y, 1, angle) for x, y, angle in inner_hex_data]
  outer_hex_params = (
      outer_hex_center[0], outer_hex_center[1],
      outer_hex_side_length, outer_hex_angle_degrees
  )

  _, ax = plt.subplots(figsize=(6, 6))
  ax.set_aspect('equal')

  # Plot outer hexagon (in red).
  outer_hex_vertices = hexagon_vertices(*outer_hex_params)
  outer_hex_x, outer_hex_y = zip(*outer_hex_vertices)
  ax.plot(
      outer_hex_x + (outer_hex_x[0],),
      outer_hex_y + (outer_hex_y[0],),
      'r-',
      label='Outer Hexagon',
  )

  # Plot inner hexagons (in blue).
  for i, inner_hex_params in enumerate(inner_hex_params_list):
    inner_hex_vertices = hexagon_vertices(*inner_hex_params)
    inner_hex_x, inner_hex_y = zip(*inner_hex_vertices)
    ax.fill(
        inner_hex_x + (inner_hex_x[0],),
        inner_hex_y + (inner_hex_y[0],),
        alpha=0.7,
        color='blue',
        label='Inner Hexagons' if i == 0 else None,  # Label only once.
    )

  ax.set_title('Hexagon Packing Construction')
  ax.legend()
  ax.grid(False)
  plt.savefig(filepath, format='png', bbox_inches='tight')
  plt.close()



class StatHexagon(Stat):
    """
    Statistical module for metaheuristic algorithms. Compute descriptive statistics and pairwise comparison and
    ranking.
    """
    def _init_params(self):
        super()._init_params()
        self._data = None

    ####################################################################
    #########  Public functions
    ####################################################################
    def evaluate_statistic(self, solutions: list[Solution]):

        self._data = {}

        best_fitness = sys.float_info.max
        best = {}
        i = 0
        for solution in solutions:
            i += 1
            if solution.get_metadata():
                if solution.get_metadata().get('results'):
                    if solution.get_metadata().get('results').get('outer_hex_side_length'):
                        self._data[str(i)] = {
                            'inner_hex_data': solution.get_metadata().get('results').get('inner_hex_data'),
                            'outer_hex_center': solution.get_metadata().get('results').get('outer_hex_center'),
                            'outer_hex_side_length': solution.get_metadata().get('results').get(
                                'outer_hex_side_length'),
                            'outer_hex_angle_degrees': solution.get_metadata().get('results').get(
                                'outer_hex_angle_degrees')
                        }
                        if solution.get_metadata().get('results').get('outer_hex_side_length') <= best_fitness:
                            best_fitness = solution.get_metadata().get('results').get('outer_hex_side_length')
                            self._data['best'] = {
                            'inner_hex_data': solution.get_metadata().get('results').get('inner_hex_data'),
                            'outer_hex_center': solution.get_metadata().get('results').get('outer_hex_center'),
                            'outer_hex_side_length': solution.get_metadata().get('results').get(
                                'outer_hex_side_length'),
                            'outer_hex_angle_degrees': solution.get_metadata().get('results').get(
                                'outer_hex_angle_degrees')
                        }



    def export(self, path: str):

        if self._data:
            for key, value in self._data.items():
                plot_construction(value['inner_hex_data'], value['outer_hex_center'], value['outer_hex_side_length'], value['outer_hex_angle_degrees'], os.path.join(path, "stat_" + self.get_short_name() + "_" + key + ".png"))

    @classmethod
    def get_short_name(cls) -> str:
        return "stat.hexagon"

    @classmethod
    def get_long_name(cls) -> str:
        return "Hexagon Containment"

    @classmethod
    def get_description(cls) -> str:
        return "Visualization of the Hexagon Containment best found solution"

    @classmethod
    def get_tags(cls) -> dict:
        return {
            'input': set(),
            'output': set()
        }

    ####################################################################
    #########  Private functions
    ####################################################################
