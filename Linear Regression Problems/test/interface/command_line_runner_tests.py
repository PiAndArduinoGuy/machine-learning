import os
import unittest

from application.interface.command_line_runner import CommandLineRunner
from test.test_utils import TestUtils


class CommandLineRunnerTests(unittest.TestCase):
    def setUp(self):
        self.commandLineRunner = CommandLineRunner()
        self.resources_location = "../resources"
        self.number_of_training_set_features = TestUtils.number_of_training_set_features
        self.training_set_size = TestUtils.training_set_size

    def test_given_an_array_of_valid_training_arguments_when_run_trainer_called_then_performs_training(self):
        valid_array_of_training_arguments = [
            self.resources_location + "/example_training_set.txt",
            self.number_of_training_set_features,
            self.training_set_size,
            TestUtils.get_initial_parameters_as_string(TestUtils.initial_parameters),
            TestUtils.learning_rate,
            TestUtils.number_of_iterations,
            self.resources_location
        ]

        self.commandLineRunner.run_trainer(valid_array_of_training_arguments)

        try:
            open(self.resources_location + "/trained_model_parameters.model")
        except FileNotFoundError:
            self.fail("The expected trained artifact trained_model_parameters.model was not present.")
        try:
            open(self.resources_location + "/cost_vs_iteration.png")
        except FileNotFoundError:
            self.fail("The expected trained artifact cost_vs_iteration.png was not present.")

        # test cleanup
        os.remove(self.resources_location + "/cost_vs_iteration.png")
        os.remove(self.resources_location + "/trained_model_parameters.model")

    def test_given_an_array_of_valid_predicting_arguments_when_run_predictor_called_then_predict(self):
        valid_array_of_predict_arguments = [
            TestUtils.get_predict_request_string(TestUtils.predict_request),
            self.resources_location + "/example_trained_parameters.model",
            self.resources_location + "/example_training_set.txt",
            self.number_of_training_set_features,
            self.training_set_size
        ]

        predicted_value = self.commandLineRunner.run_predictor(valid_array_of_predict_arguments)

        self.assertIsNotNone(predicted_value)


if __name__ == '__main__':
    unittest.main()
