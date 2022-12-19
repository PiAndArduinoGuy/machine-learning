from matplotlib import pyplot as plt
from numpy import matrix, ones, hstack, zeros, mean, std, asarray

from application.exception.model_not_trained_error import ModelNotTrainedError
from application.util.cost_function_helper import CostFunctionHelper
from application.util.gradient_descent_helper import GradientDescentHelper
from application.util.matrix_operation_helper import MatrixOperationHelper


class RegressionProblemModelTrainer:
    def __init__(self, training_x: matrix, training_y: matrix, initial_parameters: matrix, resources_location):
        """
        training_x should not contain the 0 feature values
        """

        def validate_initial_parameters(initial_parameters, num_of_features):
            if type(initial_parameters) != matrix:
                raise TypeError("The passed in parameters must be of type numpy matrix.")
            if initial_parameters.shape[1] != 1:
                raise ValueError("The parameters must be a column vector.")
            if num_of_features > initial_parameters.shape[0] - 1:
                raise ValueError("There are more features than initial parameter values provided.")
            if num_of_features < initial_parameters.shape[0] - 1:
                raise ValueError("There are more initial parameter values than features provided.")

        def validate_training_set(train_x, train_y):
            if type(train_y) != matrix:
                raise TypeError("The passed in training_y must be of type numpy matrix.")
            if type(train_x) != matrix:
                raise TypeError("The passed in training_x must be of type numpy matrix.")
            if train_y.shape[1] != 1:
                raise ValueError("The passed in training_y must be a column vector.")
            if train_x.shape[0] != train_y.shape[0]:
                raise ValueError("training_x row dim must equal training_y row dim.")

        validate_training_set(training_x, training_y)
        validate_initial_parameters(initial_parameters, num_of_features=training_x.shape[1])
        self._training_y = training_y
        self._training_x = training_x
        self._initial_parameters = initial_parameters
        self._resources_location = resources_location
        self._cost_per_iteration_list = []

    @property
    def training_y(self):
        return self._training_y

    @property
    def training_x(self):
        return self._training_x

    @training_x.setter
    def training_x(self, training_x):
        self._training_x = training_x

    @property
    def resources_location(self):
        return self._resources_location

    @property
    def initial_parameters(self):
        return self._initial_parameters

    @property
    def cost_per_iteration_list(self):
        return self._cost_per_iteration_list

    def train_model(self, learning_rate, iterations):
        self.normalize_training_x()

        training_x_with_ones_column = self.append_ones_column_to_training_x()

        next_iteration_trained_parameters = self.initial_parameters
        for i in range(iterations):
            next_iteration_trained_parameters = GradientDescentHelper.perform_gradient_descent(
                next_iteration_trained_parameters,
                training_x_with_ones_column,
                self.training_y,
                learning_rate)
            cost = CostFunctionHelper.calculate_cost_for_params(training_x_with_ones_column, self._training_y,
                                                                next_iteration_trained_parameters)
            self._cost_per_iteration_list.append(float(cost))
        return next_iteration_trained_parameters

    def append_ones_column_to_training_x(self):
        ones_column = ones((self.training_x.shape[0], 1))
        return matrix(hstack((ones_column, self.training_x)))

    def normalize_training_x(self):
        training_x_feature_means = mean(self.training_x, axis=0)
        training_x_feature_stds = std(self.training_x, axis=0)
        self.training_x = MatrixOperationHelper.normalize_matrix_using_means_and_stds(matrix=self.training_x,
                                                                                      matrix_means=training_x_feature_means,
                                                                                      matrix_stds=training_x_feature_stds)

    def save_cost_vs_iteration_plot(self):
        if len(self.cost_per_iteration_list) == 0:
            raise ModelNotTrainedError("The model has not been trained yet.")

        # matplotlib docs requirement that data plotted on axes be numpy.array
        iterations = asarray(range(1, len(self.cost_per_iteration_list) + 1))
        costs = asarray(self.cost_per_iteration_list)

        fig, ax = plt.subplots()
        ax.plot(iterations, costs)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost")

        ax.set_title("Cost vs Iteration")
        plt.savefig(self.resources_location + "/cost_vs_iteration.png")

    # expect a numpy matrix for trained_model_parameters !

    def save_trained_model(self, trained_model_parameters):
        """
        Saves the passed on trained_model_parameters matrix to a file in the form:
        1;2;3;4
        """
        with open(self.resources_location + "/trained_model_parameters.model",
                  "w") as new_trained_model_parameters_file:
            string_parameters = ""
            for parameter in trained_model_parameters:
                stripped_parameter = str(parameter).strip("[]")
                string_parameters = string_parameters + stripped_parameter + ";"
            rstrip_string_parameters = string_parameters.rstrip(";")  # remove the last semi-colon added
            new_trained_model_parameters_file.write(rstrip_string_parameters)
