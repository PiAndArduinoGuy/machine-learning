class ModelNotTrainedError(Exception):
    """
    Exception is thrown when a model is requested to predict but a model has not yet been trained.
    """
    pass
