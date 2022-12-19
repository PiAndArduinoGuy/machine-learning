import sys

from application.interface.command_line_runner import CommandLineRunner

if __name__ == "__main__":
    """    
    7 command line arguments passed - training for the requested model commences with the given command line arguments, 
    in the format
    training_file_location number_of_features number_of_training_examples initial_parameters learning_rate 
    number_of_iterations saved_resources_location
        
    5 command line arguments passed - predict the given input of features values with the given location of trained 
    model parameters, in the format
    predict_request trained_model_parameters_file_location training_set_file_location training_set_number_of_features training_set_number_of_training_examples
    
    """
    if len(sys.argv[1:]) == 7:
        commandLineRunner = CommandLineRunner()
        commandLineRunner.run_trainer(commandLineTrainingArguments=sys.argv[1:])
    elif len(sys.argv[1:]) == 5:
        commandLineRunner = CommandLineRunner()
        predicted_value = commandLineRunner.run_predictor(commandLinePredictArguments=sys.argv[1:])
        print("Predicted: ", predicted_value)
