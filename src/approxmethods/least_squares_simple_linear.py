from __future__ import annotations

import numpy as np

class LeastSquaresSimpleLinear:
    def __init__(self, x: list[float], y: list[float]):
        if len(x) != len(y):
            raise ValueError("x e y devem ter o mesmo tamanho")

        self.x = np.array(x, dtype=float)
        self.y = np.array(y, dtype=float)
        self.coefficients = self.calculate_coefficients()

    def calculate_coefficients(self):
        sum_x = np.sum(self.x)
        sum_y = np.sum(self.y)
        sum_x_squared = np.sum(self.x ** 2)
        sum_x_y = np.sum(self.x * self.y)
        n = len(self.x)

        denominator = n * sum_x_squared - sum_x ** 2
        if denominator == 0:
            raise ZeroDivisionError("Denominador zero — todos os x são iguais")

        slope = (n * sum_x_y - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n

        return {'intercept': intercept, 'slope': slope}

    def predict(self, x):
        return self.coefficients['intercept'] + self.coefficients['slope'] * np.array(x)
