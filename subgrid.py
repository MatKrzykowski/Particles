"""subgrid.py"""

import numpy as np


class Subgrid():

    def __init__(self):
        self._items = []
        self.recalculate_r_array = True
        self.r_array = None

        self.x_arr = np.zeros((1, 1))
        self.y_arr = np.zeros((1, 1))
        self._dist_arr = np.zeros((1, 1))
        self._applicable_array = np.zeros((1, 1))

    def add(self, item):
        self._items.append(item)
        self.recalculate_r_array = True

    def remove(self, item):
        self._items.remove(item)
        self.recalculate_r_array = True

    @property
    def items(self):
        return self._items

    def __bool__(self):
        return len(self._items) > 1

    def calc_r_array(self):
        if self.recalculate_r_array:
            del self.r_array
            radius_arr = np.array([i.radius for i in self._items])
            self.r_array = np.add.outer(radius_arr, radius_arr)
            self.recalculate_r_array = False
        return self.r_array

    @property
    def dist_arr(self):
        dist_arr_1d = np.array([i.position for i in self._items]).transpose()
        x_arr_1d = dist_arr_1d[0, :]
        y_arr_1d = dist_arr_1d[1, :]

        # Reuse old memory
        if self._dist_arr.shape[0] == len(self._items):
            self.x_arr = np.subtract.outer(x_arr_1d, x_arr_1d, out=self.x_arr)
            self.y_arr = np.subtract.outer(y_arr_1d, y_arr_1d, out=self.y_arr)
            self._dist_arr = np.hypot(
                self.x_arr, self.y_arr, out=self._dist_arr)

            np.subtract(
                self.calc_r_array(), self._dist_arr, out=self._applicable_array)

        # Alocate new memory
        else:
            del self.x_arr
            del self.y_arr
            del self._dist_arr
            del self._applicable_array
            self.x_arr = np.subtract.outer(x_arr_1d, x_arr_1d)
            self.y_arr = np.subtract.outer(y_arr_1d, y_arr_1d)
            self._dist_arr = np.hypot(self.x_arr, self.y_arr)

            self._applicable_array = self.calc_r_array() - self._dist_arr

        np.heaviside(self._applicable_array, 1.0, out=self._applicable_array)
        return self._dist_arr, self._applicable_array