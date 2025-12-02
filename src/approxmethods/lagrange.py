from __future__ import annotations

import numpy as np

class Lagrange:
    def __init__(self, x_list: list[float], y_list: list[float], x_predict: float):
        if len(x_list) != len(y_list):
            raise ValueError("x_list e y_list devem ter o mesmo tamanho")

        self.x = np.asarray(x_list, dtype=float)
        self.y = np.asarray(y_list, dtype=float)
        x_predict = float(x_predict)

        self.coefficients = self._compute_polynomial_coefficients()
    
    def _compute_polynomial_result(self):
        n = len(self.x)
        result = 0

        for i in range(n):
            lagrange_term = self.y[i]

            for j in range(n):
                if i != j:
                    lagrange_term *= (self.x_predict - self.x[j]) / (self.x[i] - self.x[j])

            result += lagrange_term

        return result
