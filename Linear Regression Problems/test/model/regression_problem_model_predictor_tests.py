import unittest
from unittest.mock import patch

from numpy import matrix, array_equal, hstack, std, mean

from application.models.regression_problem_model_predictor import RegressionProblemModelPredictor
from test.test_utils import TestUtils


class RegressionProblemModelPredictorTests(unittest.TestCase):
    def setUp(self):
        self.training_x = TestUtils.get_expected_training_x_matrix("../resources/example_training_set.txt")
        self.parameters = TestUtils.get_expected_trained_parameters("../resources/example_trained_parameters.model")
        self.predict_request = TestUtils.predict_request

    def test_given_trained_parameters_not_numpy_matrix_when_predict_called_then_throw_exception(self):
        with self.assertRaises(TypeError) as context:
            RegressionProblemModelPredictor(training_x=self.training_x, trained_parameters=[[1], [1], [1], [1]])
        self.assertEqual("The trained parameters '[[1], [1], [1], [1]]' is not of type numpy matrix.",
                         str(context.exception))

    def test_given_trained_parameters_not_column_vector_when_predict_called_then_throw_exception(self):
        with self.assertRaises(ValueError) as context:
            RegressionProblemModelPredictor(training_x=self.training_x, trained_parameters=matrix("1 1 1 1"))
        self.assertEqual("The parameters must be a column vector.", str(context.exception))

    @patch("application.models.regression_problem_model_predictor.matmul")
    def test_given_trained_parameters_valid_when_predict_called_then_predict_request_normalized_before_predicting(self,
                                                                                                                  mock_mat_mul):
        regression_problem_model_predictor = RegressionProblemModelPredictor(training_x=self.training_x,
                                                                             trained_parameters=self.parameters)

        regression_problem_model_predictor.predict(self.predict_request)

        model_predict_request, *other = mock_mat_mul.call_args.args
        self.assertThatPredictRequestNormalizedBeforePredicting(model_predict_request, self.training_x)

    def test_given_trained_parameters_valid_and_a_model_as_trained_when_predict_called_then_a_prediction_is_made(self):
        try:
            regression_problem_model_predictor = RegressionProblemModelPredictor(training_x=self.training_x,
                                                                                 trained_parameters=self.parameters)

            predicted_y = regression_problem_model_predictor.predict(self.predict_request)
            self.assert_expected_prediction_made(predicted_y)
        except(ValueError, TypeError):
            self.fail("There should be no exception thrown.")

    def test_given_trained_parameters_row_dim_not_equal_to_predict_request_row_dim_when_predict_called_then_throw_exception(
            self):
        regression_problem_model_predictor = RegressionProblemModelPredictor(training_x=self.training_x,
                                                                             trained_parameters=matrix("1; 1; 1"))
        with self.assertRaises(ValueError) as context:
            regression_problem_model_predictor.predict(self.predict_request)

        self.assertEqual(
            "The col dimension of the model input matrix needs to equal the row dimension of the trained parameters matrix.",
            str(context.exception))

    def test_given_input_not_numpy_matrix_when_predict_called_then_throws_exception(self):
        regression_problem_model_predictor = RegressionProblemModelPredictor(training_x=self.training_x,
                                                                             trained_parameters=self.parameters)

        with self.assertRaises(TypeError) as context:
            regression_problem_model_predictor.predict([[1], [2], [3]])
        self.assertEqual("The provided input is not of type numpy matrix.", str(context.exception))

    def test_given_input_not_row_vector_when_predict_called_then_throws_exception(self):
        regression_problem_model_predictor = RegressionProblemModelPredictor(training_x=self.training_x,
                                                                             trained_parameters=self.parameters)

        with self.assertRaises(ValueError) as context:
            regression_problem_model_predictor.predict(matrix("1 ;2; 3"))
        self.assertEqual("The provided input is not of a row vector.", str(context.exception))

    def assertThatPredictRequestNormalizedBeforePredicting(self, model_predict_request, training_x):
        training_x_feature_means = mean(training_x, axis=0)
        training_x_feature_stds = std(training_x, axis=0)
        expected_normalized_predict_request = TestUtils.get_expected_normalized_matrix(self.predict_request,
                                                                                       training_x_feature_stds,
                                                                                       training_x_feature_means)
        expected_normalized_predict_request_with_ones_feature = matrix(
            hstack(([[1]], expected_normalized_predict_request)))
        self.assertTrue(
            array_equal(expected_normalized_predict_request_with_ones_feature, model_predict_request)
        )

    @staticmethod
    def get_expected_predicted_value():
        with open("../resources/example_training_set.txt") as training_set_file:
            for training_example_line in training_set_file:
                training_example = training_example_line.rstrip().split(",")
                training_example_x = float(training_example[0])
                if training_example_x == TestUtils.predict_request[0, 0]:
                    return float(training_example[1])

    def assert_expected_prediction_made(self, predicted_y):
        self.assertIsNotNone(predicted_y)
        expected_predict_value = RegressionProblemModelPredictorTests.get_expected_predicted_value()
        allowed_error = 0.1
        if not self.is_within_allowable_expected_value_range(allowed_error, expected_predict_value, predicted_y):
            self.fail(
                f"The predicted value {predicted_y} is out of the error ranges {expected_predict_value - allowed_error} and {expected_predict_value + allowed_error}")

    @staticmethod
    def is_within_allowable_expected_value_range(allowed_error, expected_predict_value, predicted_y):
        return ((expected_predict_value - allowed_error) < predicted_y and (
                expected_predict_value + allowed_error) > predicted_y)


if __name__ == '__main__':
    unittest.main()
