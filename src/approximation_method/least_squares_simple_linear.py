from __future__ import annotations

import numpy as np

class LeastSquaresSimpleLinear:
    def __init__(self, x: list[float], y: list[float]):
        self.x = np.array(x)
        self.y = np.array(y)
        self.coefficients = self.calculate_coefficients()

    def calculate_coefficients(self):
        sum_x = np.sum(self.x)
        sum_y = np.sum(self.y)
        sum_x_squared = np.sum(self.x ** 2)
        sum_x_y = np.sum(self.x * self.y)
        n = len(self.x)

        denominator = (sum_x_squared * n - sum_x ** 2)
        slope = (sum_y * n - sum_x * sum_x_y) / denominator
        intercept = (sum_x_squared * sum_x_y - sum_x * sum_x_squared * sum_y) / denominator

        return {'intercept': intercept, 'slope': slope }
