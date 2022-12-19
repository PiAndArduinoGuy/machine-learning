from numpy import matrix, matmul, hstack, mean, std

from application.util.matrix_operation_helper import MatrixOperationHelper


class RegressionProblemModelPredictor:
    def __init__(self, training_x, trained_parameters):
        def validate_trained_parameters(trained_parameters: matrix):
            if type(trained_parameters) != matrix:
                raise TypeError(f"The trained parameters '{trained_parameters}' is not of type numpy matrix.")
            if trained_parameters.shape[1] != 1:
                raise ValueError("The parameters must be a column vector.")

        validate_trained_parameters(trained_parameters)
        self._training_x = training_x
        self._trained_parameters = trained_parameters

    @property
    def trained_parameters(self):
        return self._trained_parameters

    @property
    def training_x(self):
        return self._training_x

    def predict(self, model_input: matrix):
        def validate_model_input(model_input):
            if type(model_input) != matrix:
                raise TypeError("The provided input is not of type numpy matrix.")
            if model_input.shape[0] != 1:
                raise ValueError("The provided input is not of a row vector.")

        def validate_model_input_and_trained_parameter_dims(model_input: matrix,
                                                            trained_parameters: matrix):
            if model_input.shape[1] != trained_parameters.shape[0] - 1:
                raise ValueError(
                    "The col dimension of the model input matrix needs to equal the row dimension of the trained parameters matrix.")

        validate_model_input(model_input)
        validate_model_input_and_trained_parameter_dims(model_input, self.trained_parameters)
        normalized_model_input = MatrixOperationHelper.normalize_matrix_using_means_and_stds(matrix=model_input,
                                                                                             matrix_means=mean(self.training_x, axis=0),
                                                                                             matrix_stds=std(self.training_x, axis=0))
        normalized_model_input_with_ones_feature = matrix(hstack(([[1]], normalized_model_input)))
        prediction_matrix = matmul(normalized_model_input_with_ones_feature, self.trained_parameters)
        return prediction_matrix[0, 0]