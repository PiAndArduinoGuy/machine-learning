from numpy import zeros
from numpy import matrix


class InputFileReader:
    @staticmethod
    def get_training_set_from_file(input_file_location, training_set_size, number_of_features):
        def initialize_x_and_y_matrices(rows, columns):
            x_zeros = zeros([rows, columns])
            y_zeros = zeros([rows, 1])
            return x_zeros, y_zeros

        def get_x_and_y_for_training_example(training_item):
            features_and_output_list = training_item.rstrip().split(",")
            x = features_and_output_list[0:len(features_and_output_list) - 1]
            y = features_and_output_list[len(features_and_output_list) - 1]
            return x, y

        with open(input_file_location) as input_file:
            x_matrix, y_matrix = initialize_x_and_y_matrices(training_set_size, number_of_features)
            for index, training_item in enumerate(input_file):
                x, y = get_x_and_y_for_training_example(training_item)
                x_matrix[index, :] = x
                y_matrix[index, 0] = y
            return matrix(x_matrix), matrix(y_matrix)

    @staticmethod
    def get_trained_model_parameter_from_file(trained_model_parameters_file_location):
        with open(trained_model_parameters_file_location) as trained_model_parameters_file:
            content = trained_model_parameters_file.read()
        return matrix(content)

