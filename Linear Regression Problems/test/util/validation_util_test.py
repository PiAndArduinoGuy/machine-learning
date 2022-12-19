import unittest

from application.exception.invalid_model_name_error import InvalidModelNameError
from application.util.validation_util import ValidationUtil


class ValidationUtilTest(unittest.TestCase):
    def test_given_file_that_does_not_exist_when_validate_file_location_called_throw_exception(self):
        file_location_that_does_not_exist = "../resourcez"

        with self.assertRaises(FileNotFoundError) as context:
            ValidationUtil.validate_file_location(file_location_that_does_not_exist)
        self.assertEqual("The provided location '../resourcez' does not exist.", str(context.exception))

    def test_given_file_that_does_exists_when_validate_file_location_called_throw_no_exception(self):
        file_location_that_does_not_exist = "../resources"

        try:
            ValidationUtil.validate_file_location(file_location_that_does_not_exist)
        except FileNotFoundError:
            self.fail("A FileNotFoundError was not expected.")

    def test_given_an_action_not_part_of_recognized_actions_when_validate_predict_or_train_response_called_throw_exception(
            self):
        with self.assertRaises(ValueError) as context:
            ValidationUtil.validate_predict_or_train_response("not a recognized action.")
        self.assertEqual(
            "The provided action 'not a recognized action.' is not a recognized action, train or predict are the only valid actions.",
            str(context.exception))

    def test_given_an_action_part_of_recognized_actions_when_validate_predict_or_train_response_called_throw_no_exception(
            self):
        try:
            ValidationUtil.validate_predict_or_train_response("train")
        except ValueError:
            self.fail("A ValueError was not expected.")

    def test_given_initial_params_invalid_when_validate_initial_parameters_called_throw_exception(self):
        with self.assertRaises(ValueError) as context:
            ValidationUtil.validate_numpy_matrix("kaas")
        self.assertEqual("The provided numpy matrix 'kaas' could not be parsed as a numpy matrix.",
                         str(context.exception))

    def test_given_initial_params_valid_when_validate_initial_parameters_called_throw_no_exception(self):
        try:
            ValidationUtil.validate_numpy_matrix("1;2")
        except ValueError:
            self.fail("A ValueError was not expected.")

    def test_given_training_set_size_not_number_when_validate_training_set_size_called_throw_exception(self):
        with self.assertRaises(ValueError) as context:
            ValidationUtil.validate_is_a_number("1o")
        self.assertEqual("The provided number '1o' is not a number.", str(context.exception))

    def test_given_training_set_valid_when_validate_training_set_size_called_throw_no_exception(self):
        try:
            ValidationUtil.validate_is_a_number("10")
        except ValueError:
            self.fail("A ValueError was not expected.")

    def test_given_not_a_float_for_learning_rate_when_validate_learning_rate_called_then_throw_exception(self):
        with self.assertRaises(ValueError) as context:
            ValidationUtil.validate_learning_rate("0.o1")
        self.assertEqual("The provided learning rate '0.o1' is not a value float.", str(context.exception))

    def test_given_valid_learning_rate_when_validate_learning_rate_called_then_throw_no_exception(self):
        try:
            ValidationUtil.validate_learning_rate("0.01")
        except ValueError:
            self.fail("A ValueError was not expected.")


if __name__ == '__main__':
    unittest.main()
