# Script to train machine learning model.
import pickle as pkl

from sklearn.model_selection import train_test_split
import pandas as pd
from ml.data import process_data
from ml.model import train_model,compute_model_metrics


# load in the data from path into a pandas dataframe.
data_path = "starter/data/census_cleaned.csv"
data = pd.read_csv(data_path)



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
X_data, y_data, encoder, lb = process_data(
    data, categorical_features=cat_features, label="salary", training=True
)
print("X_data.shape: ", X_data.shape)
print("y_data.shape: ", y_data.shape)
# Optional enhancement, use K-fold cross validation instead of a train-test split.
X_train, X_test, y_train, y_test = train_test_split(X_data,y_data, test_size=0.2, random_state=2)
print("X_train.shape: ", X_train.shape)
print("y_train.shape: ", y_train.shape)

print("X_test.shape: ", X_test.shape)
print("y_test.shape: ", y_test.shape)

# Train and save a model.
model = train_model(X_train, y_train)

# evaulate model 
# create a list of predictions from the model
predictions = model.predict(X_test)
precision, recall, fbeta = compute_model_metrics(y_test, predictions)

# print metrics
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {fbeta}")

# print accuracy of model
print(f"Accuracy: {model.score(X_test, y_test)}")

# save the model as a pkl file.
model_path = "starter/model/model.pkl"
pkl.dump(model, open(model_path, "wb"))

