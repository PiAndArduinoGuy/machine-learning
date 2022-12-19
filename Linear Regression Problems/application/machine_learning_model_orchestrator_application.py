from application.models.regression_problem_model_predictor import RegressionProblemModelPredictor
from application.models.regression_problem_model_trainer import RegressionProblemModelTrainer
from application.util.input_file_reader import InputFileReader


class MachineLearningModelOrchestrator:
    @staticmethod
    def train_model(model: RegressionProblemModelTrainer, learning_rate, number_of_iterations):
        trained_model_parameters = model.train_model(learning_rate, number_of_iterations)
        model.save_cost_vs_iteration_plot()
        model.save_trained_model(trained_model_parameters)
        return model

    @staticmethod
    def create_model_trainer(training_set_file_location, number_of_training_set_features, training_set_size,
                             initial_parameters, resources_location):
        training_x, training_y = InputFileReader.get_training_set_from_file(
            input_file_location=training_set_file_location,
            number_of_features=number_of_training_set_features,
            training_set_size=training_set_size)

        model = RegressionProblemModelTrainer(training_x, training_y, initial_parameters, resources_location)

        return model

    @staticmethod
    def create_model_predictor(training_set_file_location, trained_model_parameters_file_location, number_of_training_set_features, training_set_size):
        training_x, training_y = InputFileReader.get_training_set_from_file(
            input_file_location=training_set_file_location,
            number_of_features=number_of_training_set_features,
            training_set_size=training_set_size)
        trained_parameters = InputFileReader.get_trained_model_parameter_from_file(trained_model_parameters_file_location)
        model = RegressionProblemModelPredictor(training_x, trained_parameters)

        return model

    @staticmethod
    def predict(model_predictor, predict_request):
        return model_predictor.predict(predict_request)
