from numpy import zeros, matrix, array_equal, ones, hstack, matmul, power, array2string

from application.util.input_file_reader import InputFileReader


class TestUtils:
    number_of_training_set_features = 1
    training_set_size = 1000
    initial_parameters = matrix("0;0")
    resources_location = "resources"
    learning_rate = 0.01
    number_of_iterations = 400
    predict_request = matrix("825")  # must be a value from the example_training_set.txt file

    @staticmethod
    def get_expected_normalized_matrix(matrix_to_normalize, expected_training_x_feature_stds,
                                       expected_training_x_feature_means):
        expected_normalized_matrix = zeros(matrix_to_normalize.shape)
        for col in range(matrix_to_normalize.shape[1]):
            for row in range(matrix_to_normalize.shape[0]):
                expected_normalized_matrix[row, col] = (matrix_to_normalize[row, col] -
                                                        expected_training_x_feature_means[0, col]) / \
                                                       expected_training_x_feature_stds[0, col]
        return expected_normalized_matrix

    @staticmethod
    def get_expected_training_x_matrix(training_set_file_location):
        training_x, training_y = InputFileReader.get_training_set_from_file(
            input_file_location=training_set_file_location,
            training_set_size=TestUtils.training_set_size,
            number_of_features=TestUtils.number_of_training_set_features)
        return training_x

    @staticmethod
    def get_expected_training_y_matrix(training_set_file_location):
        training_x, training_y = InputFileReader.get_training_set_from_file(
            input_file_location=training_set_file_location,
            training_set_size=TestUtils.training_set_size,
            number_of_features=TestUtils.number_of_training_set_features)
        return training_y

    @staticmethod
    def get_valid_model_trainer_constructor_arguments(training_set_file_location):
        return training_set_file_location, \
               TestUtils.number_of_training_set_features, \
               TestUtils.training_set_size, \
               TestUtils.initial_parameters, \
               TestUtils.resources_location

    @staticmethod
    def get_expected_initial_parameters():
        return TestUtils.initial_parameters

    @staticmethod
    def get_expected_trained_parameters(trained_model_parameters_file_location):
        trained_parameters = InputFileReader.get_trained_model_parameter_from_file(
            trained_model_parameters_file_location)
        return trained_parameters

    @staticmethod
    def get_training_x_with_ones(training_set_file_location):
        training_x = TestUtils.get_expected_training_x_matrix(training_set_file_location)
        ones_column = ones((training_x.shape[0], 1))
        return matrix(hstack((ones_column, training_x)))

    @staticmethod
    def get_expected_hypothesis(training_set_file_location, trained_model_parameters_file_location):
        training_x = TestUtils.get_training_x_with_ones(training_set_file_location)
        parameters = TestUtils.get_expected_trained_parameters(trained_model_parameters_file_location)
        return matmul(training_x, parameters)

    @staticmethod
    def get_expected_cost(training_set_file_location, trained_model_parameters_file_location):
        training_x = TestUtils.get_training_x_with_ones(training_set_file_location)
        hypothesis_matrix = matmul(training_x,
                                   TestUtils.get_expected_trained_parameters(trained_model_parameters_file_location))
        hypothesis_matrix_minus_training_y_squared = power(
            (hypothesis_matrix - TestUtils.get_expected_training_y_matrix(training_set_file_location)), 2)
        hypothesis_matrix_minus_training_y_squared_sum = hypothesis_matrix_minus_training_y_squared.sum()
        training_sample_size = training_x.shape[0]
        return hypothesis_matrix_minus_training_y_squared_sum / (2 * training_sample_size)

    @staticmethod
    def get_initial_parameters_as_string(initial_parameters: matrix):
        return str(initial_parameters).replace("\n", ";").replace("[", "").replace("]", "")

    @staticmethod
    def get_predict_request_string(predict_request: matrix):
        return str(predict_request).replace("[", "").replace("]", "")
