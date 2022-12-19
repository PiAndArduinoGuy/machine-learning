import unittest

from numpy import matrix, array_equal

from application.util.cost_function_helper import CostFunctionHelper
from test.test_utils import TestUtils


class CostFunctionTest(unittest.TestCase):

    def setUp(self):
        self.training_set_file_location = "../resources/example_training_set.txt"
        self.trained_model_parameters_file_location = "../resources/example_trained_parameters.model"
        self.training_x = TestUtils.get_training_x_with_ones(self.training_set_file_location)
        self.training_y = TestUtils.get_expected_training_y_matrix(self.training_set_file_location)
        self.parameters = TestUtils.get_expected_trained_parameters("../resources/example_trained_parameters.model")

    def test_given_parameter_rows_not_equal_to_training_x_columns_when_calculate_cost_for_params_called_then_throw_exception(self):
        with self.assertRaises(ValueError) as context:
            CostFunctionHelper.calculate_cost_for_params(self.training_x, self.training_y, parameters=matrix("1;2;3"))
        self.assertEqual(str(context.exception), "training_x column dim must equal 1 less than parameter's row dim.")

    def test_given_a_set_of_parameters_and_training_x_when_calculate_hypothesis_for_training_set_called_then_calculate_expected_hypothesis(self):
        hypothesis_matrix = CostFunctionHelper.calculate_hypothesis_for_training_set(self.training_x,
                                                                                     parameters=self.parameters)
        self.assertTrue(
            array_equal(hypothesis_matrix, TestUtils.get_expected_hypothesis(self.training_set_file_location,
                                                                             self.trained_model_parameters_file_location)))

    def test_given_training_x_training_y_and_parameters_when_get_expected_cost_then_calculate(self):
        cost = CostFunctionHelper.calculate_cost_for_params(self.training_x, self.training_y, self.parameters)

        expected_cost = TestUtils.get_expected_cost(self.training_set_file_location,
                                                    self.trained_model_parameters_file_location)
        self.assertEqual(expected_cost, cost)


if __name__ == '__main__':
    unittest.main()
