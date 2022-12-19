import unittest

from numpy import std, mean, matrix, ones, hstack

from application.util.cost_function_helper import CostFunctionHelper
from application.util.gradient_descent_helper import GradientDescentHelper
from test.test_utils import TestUtils


class GradientDescentHelperTest(unittest.TestCase):
    def setUp(self):
        self.training_set_file_location = "../resources/example_training_set.txt"
        self.trained_model_parameters_file_location = "../resources/example_trained_parameters.model"
        self.normalized_training_x_with_ones = self.get_expected_normalized_training_x_with_ones_column() # use normalized to ensure fastest reduction of cost when gradient descent performed
        self.initial_parameters = TestUtils.initial_parameters
        self.training_y = TestUtils.get_expected_training_y_matrix(self.training_set_file_location)
        self.learning_rate = TestUtils.learning_rate

    def test_given_valid_parameters_to_perform_gradient_descent_when_gradient_descent_called_for_one_iteration_then_new_theta_values_returned(
            self):

        new_theta = GradientDescentHelper.perform_gradient_descent(
            self.initial_parameters,
            self.normalized_training_x_with_ones,
            self.training_y,
            TestUtils.learning_rate)

        self.assertIsNotNone(new_theta)

    def test_given_5_iterations_when_perform_gradient_descent_called_then_cost_decreases(self):
        costs_per_iteration = []
        parameters = self.initial_parameters
        for i in range(5):
            new_parameters = GradientDescentHelper.perform_gradient_descent(parameters,
                                                                            self.normalized_training_x_with_ones,
                                                                            self.training_y,
                                                                            self.learning_rate)
            cost = CostFunctionHelper.calculate_cost_for_params(
                self.normalized_training_x_with_ones,
                self.training_y,
                new_parameters)
            costs_per_iteration.append(cost)
            parameters = new_parameters

        for i in range(len(costs_per_iteration) - 1):
            self.assertTrue(costs_per_iteration[i + 1] < costs_per_iteration[i])

    def get_expected_normalized_training_x_with_ones_column(self):
        training_x = TestUtils.get_expected_training_x_matrix(self.training_set_file_location)
        normalized_training_x = TestUtils.get_expected_normalized_matrix(training_x,
                                                                         std(training_x, axis=0),
                                                                         mean(training_x, axis=0)
                                                                         )
        ones_column = ones((training_x.shape[0], 1))
        return matrix(hstack((ones_column, normalized_training_x)))


if __name__ == '__main__':
    unittest.main()
