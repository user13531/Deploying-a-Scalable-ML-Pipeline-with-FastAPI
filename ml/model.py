import pickle
import numpy as np
from typing import Tuple

from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import process_data
from sklearn.base import BaseEstimator
import joblib
from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = LogisticRegression()  # Increase max_iter if convergence warnings occur
    model.fit(X_train, y_train)
    return model

def compute_model_metrics(y, preds) -> Tuple[float, float, float]:
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta

def inference(model, X) -> np.array:
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn.ensemble.GradientBoostingClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)
    
def save_model(model: BaseEstimator, path) -> None:
    """ Serializes model to a file.

    Inputs
    ------
    model
        Trained machine learning model or OneHotEncoder.
    path : str
        Path to save pickle file.
    """
    joblib.dump(model, path)
    
def load_model(path):
    """ Loads pickle file from `path` and returns it."""
    return joblib.load(path)

def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model
):
    """ Computes the model metrics on a slice of the data specified by a column name and

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    column_name : str
        Column containing the sliced feature.
    slice_value : str, int, float
        Value of the slice feature.
    categorical_features: list
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    model : sklearn.ensemble.GradientBoostingClassifier
        Model used for the task.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float

    """
  # TODO: implement the function
    X_slice, y_slice, _, _ = process_data(
        data[data[column_name] == slice_value],  # Filter the data for the slice value
        #data,
        categorical_features=categorical_features,  # Specify your categorical features if needed
        label=label,   # Replace with your actual label column name
        encoder=encoder, 
        lb=lb,
        training=False,
        )

    preds = inference(model, X_slice)

    precision, recall, fbeta = compute_model_metrics(y_slice, preds)

    return precision, recall, fbeta
