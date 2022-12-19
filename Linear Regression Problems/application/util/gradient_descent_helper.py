from numpy import matrix, zeros, transpose

from application.util.cost_function_helper import CostFunctionHelper


class GradientDescentHelper:
    @staticmethod
    def perform_gradient_descent(parameters, training_x: matrix, training_y: matrix, learning_rate):
        def get_error_matrix(parameters, training_x, training_y):
            hypothesis_matrix = CostFunctionHelper.calculate_hypothesis_for_training_set(training_x, parameters)
            return hypothesis_matrix - training_y

        def get_new_theta_value(old_theta_value,
                                error_times_feature_sum,
                                learning_rate):
            return format(old_theta_value - (learning_rate / training_set_size) * error_times_feature_sum, ".4f")

        error_matrix = get_error_matrix(parameters, training_x, training_y)

        training_set_size = training_x.shape[0]
        new_theta_matrix = zeros(parameters.shape)
        for j in range(parameters.shape[0]):  # for each parameter
            error_times_feature_sum = 0
            old_theta_value = parameters[j, 0]
            for i in range(training_x.shape[0]):  # for each training item
                x = training_x[i, j]
                error_times_feature_sum += error_matrix[i, 0] * x
            new_theta_matrix[j, 0] = get_new_theta_value(old_theta_value, error_times_feature_sum, learning_rate)
        return matrix(new_theta_matrix)
