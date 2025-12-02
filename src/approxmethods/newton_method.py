from __future__ import annotations

import numpy as np

class NewtonMethod:
    def __init__(self, x_list: list[float], y_list: list[float], x_predict: float):
        if len(x_list) != len(y_list):
            raise ValueError("x_list e y_list devem ter o mesmo tamanho")

        self.x = np.asarray(x_list, dtype=float)
        self.y = np.asarray(y_list, dtype=float)
        self.calculated_newton_method = self._calculate_newton_method()

    def _calculate_newton_method(self):
        n = len(self.x)

        coef = self._calculate_newton_coefficients()
        result = coef[0]

        for i in range(1, n):
            term = coef[i]
            for j in range(i):
                term *= (self.x_predict - self.x[j])

            result += term

        return self.calculated_divided_differences

    def divided_differences(self):
        n = len(self.x_list)
        coefficients = np.zeros((n, n))
        coefficients[:, 0] = self.y_list

        for i in range(n):
            for j in range(1, n):
                coefficients[i, j] = (coefficients[i, j - 1] - coefficients[i - 1, j - 1]) / (self.x[i] - self.x[i - j])
        return coefficients


