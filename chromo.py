from enum import Enum
import numpy as np
from typing import Self, Any


class DatType(Enum):
    INTEGER = 1
    REAL = 2
    COMPLEX = 3


class Chromo(object):
    def __init__(self, dims: int, domain: DatType, low_b: int | float = 0.0, upp_b: int | float = 1.0, init: bool = True) -> None:
        self.dims = dims
        self.domain = domain
        self.fitness = None
        self.init = init
        match domain:
            case DatType.INTEGER:
                self.lower_bound = int(low_b)
                self.upper_bound = int(upp_b) + 1
                self.matrix = np.zeros((self.dims, self.dims), dtype=int)
            case DatType.REAL:
                self.lower_bound = low_b
                self.upper_bound = upp_b
                self.matrix = np.zeros((self.dims, self.dims), dtype=float)
            case DatType.COMPLEX:
                pass
        if self.init:
            self._initialise()

    def _initialise(self) -> None:
        match self.domain:
            case DatType.INTEGER:
                tmp_mat = np.array(np.random.randint(self.lower_bound, self.upper_bound, (self.dims, self.dims)), dtype=int)
            case DatType.REAL:
                tmp_mat = np.array(np.random.uniform(self.lower_bound, self.upper_bound, (self.dims, self.dims)), dtype=float)
            case _:
                tmp_mat = None
        for i in range(self.dims):
            tmp_mat[i, i] = 0.0
        self.matrix = tmp_mat

    def _swap_cols(self, i: int, j: int) -> None:
        self.matrix[:, [i, j]] = self.matrix[:, [j, i]]

    def _swap_rows(self, i: int, j: int) -> None:
        self.matrix[[i, j], :] = self.matrix[[j, i], :]

    def _swap_elems(self, i1: int, j1: int, i2: int, j2: int) -> None:
        self.matrix[i1, j1], self.matrix[i2, j2] = self.matrix[i2, j2], self.matrix[i1, j1]

    def x_over_cols(self, other: Self, col: int) -> tuple[Any, Any]:
        if (1 <= col) and (col < self.dims):
            left = np.concatenate((self.matrix[:, list(range(col + 1))], other.matrix[:, list(range(col + 1, self.dims, 1))]), axis=1)
            right = np.concatenate((other.matrix[:, list(range(col + 1))], self.matrix[:, list(range(col + 1, self.dims, 1))]), axis=1)
        elif col < 1:
            left = self.matrix
            right = other.matrix
        else:
            left = other.matrix
            right = self.matrix
        # for i in range(self.dims):
        #     left[i, i] = 0
        #     right[i, i] = 0

        return left, right

    def x_over_rows(self, other: Self, row: int) -> tuple[Any, Any]:

        if (1 <= row) and (row < self.dims):
            above = np.concatenate((self.matrix[list(range(row + 1)), :], other.matrix[list(range(row + 1, self.dims, 1)), :]), axis=0)
            below = np.concatenate((other.matrix[list(range(row + 1)), :], self.matrix[list(range(row + 1, self.dims, 1)), :]), axis=0)
        elif row < 1:
            above = self.matrix
            below = other.matrix
        else:
            above = other.matrix
            below = self.matrix
        # for i in range(self.dims):
        #     above[i, i] = 0
        #     below[i, i] = 0
        return above, below

    def __repr__(self) -> str:
        return str(self.matrix)


if __name__ == '__main__':
    a = Chromo(dims=5, domain=DatType.REAL, low_b=0.0, upp_b=1)
    b = Chromo(dims=5, domain=DatType.REAL, low_b=0.0, upp_b=1)
    print('a =', a)
    print('b =', b)
    c, d = a.x_over_cols(b,2)
    print('c =', c)
    print('d =', d)
    e, f = a.x_over_rows(b,2)
    print('e =', e)
    print('f =', f)

