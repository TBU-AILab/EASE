from .stat import Stat
from ..solutions.solution import Solution
import numpy as np
import pandas as pd
from scipy.stats import ranksums
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def _plot_points_in_triangle(fitness: float, points: np.ndarray, filepath: str):
    """
    Plots the points and the outer equilateral triangle, and saves the figure to a PNG file.

    Parameters:
    - points: np.ndarray of shape (n, 2) – coordinates of the points inside the triangle
    - filepath: str – path to save the output PNG image
    """
    a = np.array([0, 0])
    b = np.array([1, 0])
    c = np.array([0.5, np.sqrt(3) / 2])
    triangle_vertices = np.array([a, b, c])

    _, ax = plt.subplots(1, figsize=(5, 5))
    ax.set_aspect('equal')

    # Plot the equilateral triangle
    triangle_patch = patches.Polygon(triangle_vertices, closed=True, edgecolor='black', facecolor='none')
    ax.add_patch(triangle_patch)

    # Plot points inside the triangle
    ax.scatter(points[:, 0], points[:, 1], color='blue')

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.0)
    ax.set_title(f"Heilbronn triangles solution for n={len(points[:, 0])}, fitness={fitness}")
    ax.set_axis_off()

    plt.grid(False)
    plt.savefig(filepath, format='png', bbox_inches='tight')
    plt.close()


class StatHeilbronn(Stat):
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

        best_fitness = -1
        for solution in solutions:
            if solution.get_metadata():
                if solution.get_metadata().get('results'):
                    if solution.get_metadata().get('results').get('fitness'):
                        if solution.get_metadata().get('results').get('fitness') >= best_fitness:
                            self._data = {
                                'coordinates': solution.get_metadata().get('results').get('coordinates'),
                                'fitness': solution.get_metadata().get('results').get('fitness')
                            }
                            best_fitness = solution.get_metadata().get('results').get('fitness')


    def export(self, path: str):

        if self._data:
            _plot_points_in_triangle(self._data['fitness'], self._data['coordinates'], os.path.join(path, "stat_" + self.get_short_name() + ".png"))

    @classmethod
    def get_short_name(cls) -> str:
        return "stat.heilbronn"

    @classmethod
    def get_long_name(cls) -> str:
        return "Heilbronn Triangles"

    @classmethod
    def get_description(cls) -> str:
        return "Visualization of the Heilbronn Triangles best found solution"

    @classmethod
    def get_tags(cls) -> dict:
        return {
            'input': {'heilbronn'},
            'output': set()
        }

    ####################################################################
    #########  Private functions
    ####################################################################
