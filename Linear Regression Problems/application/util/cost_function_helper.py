from numpy import matrix
from numpy import matmul
from numpy import power


class CostFunctionHelper:

    @staticmethod
    def calculate_cost_for_params(training_x: matrix, training_y: matrix, parameters: matrix):
        def validate_parameters_dimensions(parameters, num_of_features):
            if num_of_features != parameters.shape[0]:
                raise ValueError("training_x column dim must equal 1 less than parameter's row dim.")

        validate_parameters_dimensions(parameters, num_of_features=training_x.shape[1])
        hypothesis_matrix = CostFunctionHelper.calculate_hypothesis_for_training_set(training_x, parameters)
        hypothesis_matrix_minus_training_y_squared = power((hypothesis_matrix - training_y), 2)
        hypothesis_matrix_minus_training_y_squared_sum = hypothesis_matrix_minus_training_y_squared.sum()
        training_sample_size = training_x.shape[0]
        cost = hypothesis_matrix_minus_training_y_squared_sum / (2 * training_sample_size)
        return cost

    @staticmethod
    def calculate_hypothesis_for_training_set(training_x: matrix, parameters: matrix):
        return matmul(training_x, parameters)  # h = X * theta
