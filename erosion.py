import numpy as np

class Erosion:
    def __init__(self, binary_img_matrix, structuring_element=np.ones((3, 3))):
        self.binary_img_matrix = np.asarray(binary_img_matrix)
        self.structuring_element = np.asarray(structuring_element)
        self.ste_shp = structuring_element.shape
        self.eroded_img = np.zeros((binary_img_matrix.shape[0], binary_img_matrix.shape[1]))
        self.ste_origin = (int(np.ceil((structuring_element.shape[0] - 1) / 2.0)),
                           int(np.ceil((structuring_element.shape[1] - 1) / 2.0)))

    @staticmethod
    def idx_check(index):
        if index < 0:
            return 0
        else:
            return index

    def run(self):
        for i in range(len(self.binary_img_matrix)):
            for j in range(len(self.binary_img_matrix[0])):
                overlap = self.binary_img_matrix[self.idx_check(i - self.ste_origin[0]):i + (self.ste_shp[0] - self.ste_origin[0]),
                          self.idx_check(j - self.ste_origin[1]):j + (self.ste_shp[1] - self.ste_origin[1])]

                shp = overlap.shape

                ste_first_row_idx = int(np.fabs(i - self.ste_origin[0])) if i - self.ste_origin[0] < 0 else 0
                ste_first_col_idx = int(np.fabs(j - self.ste_origin[1])) if j - self.ste_origin[1] < 0 else 0

                ste_last_row_idx = self.ste_shp[0] - 1 - (
                            i + (self.ste_shp[0] - self.ste_origin[0]) - self.binary_img_matrix.shape[0]) if i + (
                            self.ste_shp[0] - self.ste_origin[0]) > self.binary_img_matrix.shape[0] else self.ste_shp[0] - 1
                ste_last_col_idx = self.ste_shp[1] - 1 - (
                            j + (self.ste_shp[1] - self.ste_origin[1]) - self.binary_img_matrix.shape[1]) if j + (
                            self.ste_shp[1] - self.ste_origin[1]) > self.binary_img_matrix.shape[1] else self.ste_shp[1] - 1

                if shp[0] != 0 and shp[1] != 0 and np.array_equal(
                        np.logical_and(overlap, self.structuring_element[ste_first_row_idx:ste_last_row_idx + 1, ste_first_col_idx:ste_last_col_idx + 1]),
                        self.structuring_element[ste_first_row_idx:ste_last_row_idx + 1, ste_first_col_idx:ste_last_col_idx + 1]):
                    self.eroded_img[i, j] = 1

        output = (self.eroded_img * 255).astype("uint8")

        return output