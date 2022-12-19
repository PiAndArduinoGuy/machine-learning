from numpy import zeros


class MatrixOperationHelper:
    @staticmethod
    def normalize_matrix_using_means_and_stds(matrix, matrix_means, matrix_stds):
        normalized_matrix = zeros(matrix.shape)
        for col in range(matrix.shape[1]):
            for row in range(matrix.shape[0]):
                normalized_matrix[row, col] = (matrix[row, col] - matrix_means[0, col]) / matrix_stds[0, col]
        return normalized_matrix
