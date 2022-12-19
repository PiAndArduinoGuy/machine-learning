import os
import unittest
from unittest.mock import patch, PropertyMock

from numpy import matrix, array_equal, std, mean

from application.exception.model_not_trained_error import ModelNotTrainedError
from application.models.regression_problem_model_trainer import RegressionProblemModelTrainer
from test.test_utils import TestUtils


class RegressionProblemModelTrainerTests(unittest.TestCase):
    def setUp(self):
        self.training_set_file_location = "../resources/example_training_set.txt"
        self.trained_model_parameters_file_location = "../resources/example_trained_parameters.model"
        self.training_x = TestUtils.get_expected_training_x_matrix(self.training_set_file_location)
        self.training_y = TestUtils.get_expected_training_y_matrix(self.training_set_file_location)
        self.initial_parameters = TestUtils.initial_parameters
        self.resources_location = "../resources"
        self.learning_rate = TestUtils.learning_rate
        self.number_of_iterations = TestUtils.number_of_iterations
    def test_given_training_y_not_numpy_matrix_when_regression_problem_model_trainer_created_then_throw_exception(self):
        with self.assertRaises(TypeError) as context:
            RegressionProblemModelTrainer(training_x=self.training_x,
                                          training_y=[5],
                                          initial_parameters=self.initial_parameters,
                                          resources_location=self.resources_location)
        self.assertEqual(str(context.exception), "The passed in training_y must be of type numpy matrix.")

    def test_given_training_x_not_numpy_matrix_when_regression_problem_model_trainer_created_then_throw_exception(self):
        with self.assertRaises(TypeError) as context:
            RegressionProblemModelTrainer(training_x=
                                          [[6, 7],
                                           [9, 10],
                                           [12, 13]],
                                          training_y=self.training_y,
                                          initial_parameters=self.initial_parameters,
                                          resources_location=self.resources_location)
        self.assertEqual(str(context.exception), "The passed in training_x must be of type numpy matrix.")

    def test_given_training_y_not_a_column_vector_when_regression_problem_model_trainer_created_then_throw_exception(self):
        with self.assertRaises(ValueError) as context:
            RegressionProblemModelTrainer(training_x=self.training_x,
                                          training_y=matrix("1 2 3"),
                                          initial_parameters=self.initial_parameters,
                                          resources_location=self.resources_location)
        self.assertEqual("The passed in training_y must be a column vector.", str(context.exception))

    def test_given_training_x_row_dim_not_equal_to_training_y_col_dim_when_regression_problem_model_trainer_created_then_throw_exception(
            self):
        with self.assertRaises(ValueError) as context:
            RegressionProblemModelTrainer(training_x=matrix("6 7; 9 10"),
                                          training_y=self.training_y,
                                          initial_parameters=self.initial_parameters,
                                          resources_location=self.resources_location)
        self.assertEqual(str(context.exception), "training_x row dim must equal training_y row dim.")

    def test_given_training_x_training_y_valid_when_regression_problem_model_trainer_created_then_set_training_x_training_y(
            self):
        try:
            regression_problem_model = RegressionProblemModelTrainer(training_x=self.training_x,
                                                                     training_y=self.training_y,
                                                                     initial_parameters=self.initial_parameters,
                                                                     resources_location=self.resources_location)
            self.assertTrue(array_equal(regression_problem_model.training_y, TestUtils.get_expected_training_y_matrix(self.training_set_file_location)))
            self.assertTrue(array_equal(regression_problem_model.training_x, TestUtils.get_expected_training_x_matrix(self.training_set_file_location)))
        except (ValueError, TypeError):
            self.fail("There should be no exception thrown.")

    def test_given_initial_parameters_not_a_column_vector_when_regression_problem_model_trainer_created_then_throw_exception(
            self):
        with self.assertRaises(ValueError) as context:
            RegressionProblemModelTrainer(training_x=self.training_x,
                                          training_y=self.training_y,
                                          initial_parameters=matrix("1 2 3"),
                                          resources_location=self.resources_location)
        self.assertEqual(str(context.exception), "The parameters must be a column vector.")

    def test_given_initial_parameters_more_than_features_in_training_set_when_regression_problem_model_trainer_created_then_throw_exception(
            self):
        with self.assertRaises(ValueError) as context:
            RegressionProblemModelTrainer(training_x=self.training_x,
                                          training_y=self.training_y,
                                          initial_parameters=matrix("1; 2; 3; 4; 5"),
                                          resources_location=self.resources_location)
        self.assertEqual("There are more initial parameter values than features provided.", str(context.exception))

    def test_given_features_in_training_set_more_than_initial_values_when_regression_problem_model_trainer_created_then_throw_exception(
            self):
        with self.assertRaises(ValueError) as context:
            RegressionProblemModelTrainer(training_x=self.training_x,
                                          training_y=self.training_y,
                                          initial_parameters=matrix("1"),
                                          resources_location=self.resources_location)
        self.assertEqual("There are more features than initial parameter values provided.", str(context.exception))

    def test_given_type_valid_initial_parameters_when_regression_problem_model_trainer_created_then_sets_initial_parameters(
            self):
        try:
            regression_problem_model = RegressionProblemModelTrainer(training_x=self.training_x,
                                                                     training_y=self.training_y,
                                                                     initial_parameters=self.initial_parameters,
                                                                     resources_location=self.resources_location)
            self.assertTrue(array_equal(regression_problem_model.initial_parameters, self.initial_parameters))
        except (ValueError, TypeError):
            self.fail("There should be no exception thrown.")

    def test_given_valid_attributes_when_train_model_called_trained_parameters_returned(self):
        regression_problem_model = self.create_valid_regression_problem_model_trainer()

        trained_parameters = regression_problem_model.train_model(learning_rate=self.learning_rate,
                                                                  iterations=self.number_of_iterations)

        self.assertIsNotNone(trained_parameters)

    def test_given_valid_attributes_when_train_model_called_then_cost_per_iteration_calculated(self):
        regression_problem_model_trainer = self.create_valid_regression_problem_model_trainer()

        regression_problem_model_trainer.train_model(learning_rate=self.learning_rate, iterations=self.number_of_iterations)

        self.assertFalse(len(regression_problem_model_trainer.cost_per_iteration_list) == 0,
                         "When the model is trained, _cost_per_iteration_list is not suppose to be empty.")
        self.assertTrue(len(regression_problem_model_trainer.cost_per_iteration_list) == self.number_of_iterations,
                        "For 1000 iterations, there are suppose to be 100 costs.")
        for cost in enumerate(regression_problem_model_trainer.cost_per_iteration_list):
            self.assertTrue(float, type(cost))

    def test_given_a_regression_problem_model_trainer_when_normalize_training_called_then_normalize_training_x(self):
        regression_problem_model_trainer = self.create_valid_regression_problem_model_trainer()

        regression_problem_model_trainer.normalize_training_x()

        self.assert_that_training_x_normalized(regression_problem_model_trainer)

    def test_given_a_regression_problem_model_trainer_when_append_ones_column_to_training_x_called_the_training_x_updated_as_expected(self):
        regression_problem_model_trainer = self.create_valid_regression_problem_model_trainer()

        training_x_with_appended_ones_column = regression_problem_model_trainer.append_ones_column_to_training_x()
        self.assertTrue(array_equal(TestUtils.get_training_x_with_ones(self.training_set_file_location),
                                    training_x_with_appended_ones_column))

    def test_given_a_regression_problem_model_trainer_and_cost_per_iteration_list_empty_when_save_cost_vs_iteration_plot_called_then_exception_is_thrown(self):
        regression_problem_model_trainer = self.create_valid_regression_problem_model_trainer()

        with self.assertRaises(ModelNotTrainedError) as context:
            regression_problem_model_trainer.save_cost_vs_iteration_plot()
        self.assertEqual(
            "The model has not been trained yet.",
            str(context.exception))

    @patch("application.models.regression_problem_model_trainer.RegressionProblemModelTrainer.cost_per_iteration_list", new_callable=PropertyMock)
    def test_given_a_regression_problem_model_trainer_and_cost_per_iteration_list_not_empty_when_save_cost_vs_iteration_plot_called_then_no_exception_is_thrown_and_plot_saved(
            self, mock_cost_per_iteration_list):
        regression_problem_model_trainer = self.create_valid_regression_problem_model_trainer()
        mock_cost_per_iteration_list.return_value = [1, 2, 3]

        regression_problem_model_trainer.save_cost_vs_iteration_plot()

        self.assert_expected_cost_vs_iteration_file_saved()

        os.remove(self.resources_location + "/cost_vs_iteration.png")

    def test_given_valid_trained_model_parameters_when_save_trained_model_parameters_called_then_save_parameters_in_expected_format_in_expected_location(
            self):
        regression_problem_model_trainer = self.create_valid_regression_problem_model_trainer()

        regression_problem_model_trainer.save_trained_model(trained_model_parameters=TestUtils.get_expected_trained_parameters(self.trained_model_parameters_file_location))

        self.assert_expected_trained_model_parameters_file_saved()

        os.remove(self.resources_location + "/trained_model_parameters.model")

    def assert_expected_trained_model_parameters_file_saved(self):
        try:
            with open(self.resources_location + "/trained_model_parameters.model") as file:
                content = file.read()
                with open(self.resources_location + "/example_trained_parameters.model") as trained_parameters_file:
                    self.assertEqual(trained_parameters_file.read(), content)
        except FileNotFoundError:
            self.fail(f"The expected file with '../resources/trained_model_parameters.model' does not exist.")

    def create_valid_regression_problem_model_trainer(self):
        regression_problem_model = RegressionProblemModelTrainer(self.training_x,
                                                                 self.training_y,
                                                                 self.initial_parameters,
                                                                 resources_location=self.resources_location)
        return regression_problem_model

    def assert_that_training_x_normalized(self, model):
        expected_training_x_feature_means = mean(self.training_x, axis=0)
        expected_training_x_feature_stds = std(self.training_x, axis=0)

        expected_normalized_training_x_values = TestUtils.get_expected_normalized_matrix(self.training_x,
                                                                                         expected_training_x_feature_stds,
                                                                                         expected_training_x_feature_means)
        self.assertTrue(
            array_equal(expected_normalized_training_x_values, model.training_x)
        )

    def assert_expected_cost_vs_iteration_file_saved(self):
        try:
            open(self.resources_location + "/cost_vs_iteration.png")
        except FileNotFoundError:
            self.fail("The expected graph was not saved.")

if __name__ == '__main__':
    unittest.main()
