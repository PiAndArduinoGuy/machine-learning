import os
import unittest

from numpy import matrix, array_equal

from application.machine_learning_model_orchestrator_application import MachineLearningModelOrchestrator
from application.models.regression_problem_model_predictor import RegressionProblemModelPredictor
from application.models.regression_problem_model_trainer import RegressionProblemModelTrainer
from test.test_utils import TestUtils


class MachineLearningModelOrchestratorApplicationTests(unittest.TestCase):

    def setUp(self):
        self.training_set_file_location = "resources/example_training_set.txt"
        self.trained_model_parameters_file_location = "resources/example_trained_parameters.model"
        self.number_of_training_set_features = TestUtils.number_of_training_set_features
        self.training_set_size = TestUtils.training_set_size
        self.initial_parameters = TestUtils.initial_parameters
        self.resources_location = TestUtils.resources_location

    def test_given_valid_input_when_create_model_trainer_called_then_expected_model_returned(self):
        model = MachineLearningModelOrchestrator.create_model_trainer(self.training_set_file_location,
                                                                      self.number_of_training_set_features,
                                                                      self.training_set_size,
                                                                      self.initial_parameters,
                                                                      self.resources_location)

        self.assert_expected_regression_problem_model_trainer(model)

    def test_given_model_trainer_when_train_model_method_called_then_model_trained_as_expected(
            self):
        model_trainer = MachineLearningModelOrchestrator.create_model_trainer(self.training_set_file_location,
                                                                              self.number_of_training_set_features,
                                                                              self.training_set_size,
                                                                              self.initial_parameters,
                                                                              self.resources_location)

        trained_model = MachineLearningModelOrchestrator.train_model(model_trainer,
                                                                     TestUtils.learning_rate,
                                                                     TestUtils.number_of_iterations)

        self.assertTrue(TestUtils.number_of_iterations, len(trained_model.cost_per_iteration_list))

        # clean up
        os.remove("resources/cost_vs_iteration.png")
        os.remove("resources/trained_model_parameters.model")

    def test_given_valid_input_when_create_model_predictor_called_then_expected_model_returned(self):
        model = MachineLearningModelOrchestrator.create_model_predictor(self.training_set_file_location,
                                                                        self.trained_model_parameters_file_location,
                                                                        self.number_of_training_set_features,
                                                                        self.training_set_size)

        self.assert_expected_regression_problem_model_predictor(model)

    def test_given_model_predictor_when_predict_method_called_then_predict_as_expected(
            self):
        model_predictor = MachineLearningModelOrchestrator.create_model_predictor(self.training_set_file_location,
                                                                                  self.trained_model_parameters_file_location,
                                                                                  self.number_of_training_set_features,
                                                                                  self.training_set_size)

        predicted_value = MachineLearningModelOrchestrator.predict(model_predictor,
                                                                   TestUtils.predict_request)

        self.assertIsNotNone(predicted_value)

    def mock_bad_input_for_learning_rate_question(self, request_string):
        mock_values = {
            "What should the learning rate be for training?:": "0.o1",
            "How many iterations must be applied?:": 100
        }
        try:
            return mock_values[request_string]
        except KeyError:
            self.fail(
                f"A KeyError picked up invoking the side_effect method, '{request_string}' was passed in to the input mock but no such question string was expected")

    def assert_expected_regression_problem_model_trainer(self, model):
        training_set_file_location = "resources/example_training_set.txt"
        self.assertIsNotNone(model)
        self.assertEqual(RegressionProblemModelTrainer, type(model))
        self.assertIsNotNone(model.training_x)
        self.assertTrue(
            array_equal(TestUtils.get_expected_training_x_matrix(training_set_file_location), model.training_x))
        self.assertIsNotNone(model.training_y)
        self.assertTrue(
            array_equal(TestUtils.get_expected_training_y_matrix(training_set_file_location), model.training_y))
        self.assertIsNotNone(model.initial_parameters)
        self.assertTrue(array_equal(TestUtils.get_expected_initial_parameters(), model.initial_parameters))
        self.assertIsNotNone(model.cost_per_iteration_list)
        self.assertEqual([], model.cost_per_iteration_list)

    def assert_expected_regression_problem_model_predictor(self, model):
        training_set_file_location = "resources/example_training_set.txt"
        trained_model_parameters_file_location = "resources/example_trained_parameters.model"
        self.assertIsNotNone(model)
        self.assertEqual(RegressionProblemModelPredictor, type(model))
        self.assertIsNotNone(model.training_x)
        self.assertTrue(
            array_equal(TestUtils.get_expected_training_x_matrix(training_set_file_location), model.training_x))
        self.assertIsNotNone(model.trained_parameters)
        self.assertTrue(array_equal(TestUtils.get_expected_trained_parameters(trained_model_parameters_file_location),
                                    model.trained_parameters))


if __name__ == '__main__':
    unittest.main()
