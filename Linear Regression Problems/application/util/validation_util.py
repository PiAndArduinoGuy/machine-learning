from os import path

from numpy import matrix

from application.exception.invalid_model_name_error import InvalidModelNameError


class ValidationUtil:
    @staticmethod
    def validate_file_location(file_location):
        if not path.exists(file_location):
            raise FileNotFoundError(f"The provided location '{file_location}' does not exist.")

    @staticmethod
    def validate_predict_or_train_response(action):
        recognized_actions = ["train", "predict"]
        if action not in recognized_actions:
            raise ValueError(
                f"The provided action '{action}' is not a recognized action, train or predict are the only valid actions.")

    @staticmethod
    def validate_numpy_matrix(numpy_matrix):
        try:
            initial_parameters = matrix(numpy_matrix)
        except ValueError:
            raise ValueError(
                f"The provided numpy matrix '{numpy_matrix}' could not be parsed as a numpy matrix.")

    @staticmethod
    def validate_is_a_number(number):
        try:
            training_set_size = int(number)
        except ValueError:
            raise ValueError(f"The provided number '{number}' is not a number.")

    @staticmethod
    def validate_learning_rate(learning_rate):
        try:
            learning_rate = float(learning_rate)
        except ValueError:
            raise ValueError(f"The provided learning rate '{learning_rate}' is not a value float.")