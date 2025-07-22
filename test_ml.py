import pytest
import numpy as np
#from sklearn.datasets import make_classification
#from sklearn.model_selection import train_test_split
from ml.model import train_model#, #apply_labels, compute_model_metrics
# TODO: add necessary import

X_train = np.random.rand(100, 10)  # Example training data
y_train = np.random.randint(0, 2, size=100)  # Example

# TODO: implement the first test. Change the function name and input as needed
def test_train_model():
    """
    # add description for the first test
    """    
    model = train_model(X_train, y_train)
    assert model is not None, "Model should not be None"
    assert hasattr(model, 'predict'), "Trained model should have a predict method"
    pass


# TODO: implement the second test. Change the function name and input as needed
#def test_apply_labels():
    """
    # add description for the second test
    """
    #predictions = np.array([0, 1, 0, 1])
    #labels = apply_labels(predictions)
    #assert len(labels) == len(predictions), "Labels length should match predictions length"
    #assert set(labels) == {">50K", "<50K"}, "Labels should be correctly applied"
    #pass


# TODO: implement the third test. Change the function name and input as needed
#def test_compute_model_metrics():
    """
    # add description for the third test
    """
    #model = train_model(X_train, y_train)
    #y_pred = model.predict(X_test)
    #metrics = compute_model_metrics(y_test, y_pred)
    #pass

    #assert 'accuracy' in metrics, "Metrics should include accuracy"
    #assert 'precision' in metrics, "Metrics should include precision"
    #assert 'recall' in metrics, "Metrics should include recall"
    #assert metrics['accuracy'] >= 0 and metrics['accuracy'] <= 1, "Accuracy should be between 0 and 1"