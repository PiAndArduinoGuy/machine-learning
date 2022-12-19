from numpy import matrix

from application.machine_learning_model_orchestrator_application import MachineLearningModelOrchestrator
from application.util.validation_util import ValidationUtil


class CommandLineRunner:

    def run_trainer(self, commandLineTrainingArguments):
        training_file_location, \
        number_of_features, \
        number_of_training_examples, \
        initial_parameters, \
        learning_rate, \
        number_of_iterations, \
        saved_resources_location = tuple(commandLineTrainingArguments)

        CommandLineRunner.validate_training_command_line_argument_values(training_file_location,
                                                                         number_of_features,
                                                                         number_of_training_examples,
                                                                         initial_parameters,
                                                                         learning_rate,
                                                                         number_of_iterations,
                                                                         saved_resources_location)

        number_of_features = int(number_of_features)
        number_of_training_examples = int(number_of_training_examples)
        initial_parameters = matrix(initial_parameters)
        learning_rate = float(learning_rate)
        number_of_iterations = int(number_of_iterations)
        model = MachineLearningModelOrchestrator.create_model_trainer(training_file_location,
                                                                      number_of_features,
                                                                      number_of_training_examples,
                                                                      initial_parameters,
                                                                      saved_resources_location)

        MachineLearningModelOrchestrator.train_model(model, learning_rate, number_of_iterations)

    def run_predictor(self, commandLinePredictArguments):
        predict_request, \
        trained_model_parameters_file_location, \
        training_set_file_location, \
        training_set_number_of_features, \
        training_set_size = tuple(commandLinePredictArguments)

        CommandLineRunner.validate_predicting_command_line_argument_values(predict_request,
                                                                           trained_model_parameters_file_location,
                                                                           training_set_number_of_features,
                                                                           training_set_size)

        predict_request = matrix(predict_request)
        training_set_number_of_features = int(training_set_number_of_features)
        training_set_size = int(training_set_size)
        model_predictor = MachineLearningModelOrchestrator.create_model_predictor(
            training_set_file_location=training_set_file_location,
            trained_model_parameters_file_location=trained_model_parameters_file_location,
            number_of_training_set_features=training_set_number_of_features,
            training_set_size=training_set_size)

        predicted_value = model_predictor.predict(predict_request)
        return predicted_value

    @staticmethod
    def validate_training_command_line_argument_values(training_file_location, number_of_features,
                                                       number_of_training_examples, initial_parameters, learning_rate,
                                                       number_of_iterations, saved_resources_location):
        ValidationUtil.validate_file_location(training_file_location)
        ValidationUtil.validate_is_a_number(number_of_features)
        ValidationUtil.validate_is_a_number(number_of_training_examples)
        ValidationUtil.validate_learning_rate(learning_rate)
        ValidationUtil.validate_is_a_number(number_of_iterations)
        ValidationUtil.validate_numpy_matrix(initial_parameters)
        ValidationUtil.validate_file_location(saved_resources_location)

    @staticmethod
    def validate_predicting_command_line_argument_values(predict_request,
                                                         trained_model_parameters_file_location,
                                                         training_set_number_of_features,
                                                         training_set_size):
        ValidationUtil.validate_file_location(trained_model_parameters_file_location)
        ValidationUtil.validate_numpy_matrix(predict_request)
        ValidationUtil.validate_is_a_number(training_set_number_of_features)
        ValidationUtil.validate_is_a_number(training_set_size)
