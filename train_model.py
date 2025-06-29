import os

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from config import project_path, model_path, encoder_path
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# TODO: load the cencus.csv data
def get_census_data() -> DataFrame:
    data_path = os.path.join(project_path, "data", "census.csv")
    print(f"Path to the census data: {data_path}'")
    data = DataFrame = pd.read_csv(data_path, header=0)
    return data

def get_train_test_sets(data: DataFrame):
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    return train, test

data = get_census_data()
train, test = get_train_test_sets(data)

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# TODO: use the process_data function provided to process the data.
X_train, y_train, encoder, lb = process_data(
    X=train,
    categorical_features=cat_features,
    label="salary",
    training=True,
    encoder=None,
    lb=None,
)

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

np.save(os.path.join(project_path, "data", "X_test.npy"), X_test)
np.save(os.path.join(project_path, "data", "y_test.npy"), y_test)

model = train_model(X_train, y_train)

# save the model and the encoder
model_path = os.path.join(project_path, "model", "model.pkl")
save_model(model, model_path)
encoder_path = os.path.join(project_path, "model", "encoder.pkl")
save_model(encoder, encoder_path)

# load the model
model = load_model(
    model_path
) 

# TODO: use the inference function to run the model inferences on the test dataset.
preds = inference(model, X_test)

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# TODO: compute the performance on model slices using the performance_on_categorical_slice function
# iterate through the categorical features
for col in cat_features:
    # iterate through the unique values in one categorical feature
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]
        p, r, fb = performance_on_categorical_slice(
            data,
            column_name=col,
            slice_value=slicevalue,
            categorical_features=cat_features,
            label="salary",
            encoder=encoder,
            lb=lb,
            model=model,
        )
        with open("slice_output.txt", "a") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)
