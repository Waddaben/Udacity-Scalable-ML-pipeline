# Script to train machine learning model.
import pickle as pkl

from sklearn.model_selection import train_test_split
import pandas as pd
from ml.data import process_data
from ml.model import (
    train_model,
    compute_model_metrics,
    train_and_test_on_slices,
    print_metrics,
)

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

# load in the data from path into a pandas dataframe.
data_path = "starter/data/census_cleaned.csv"
data = pd.read_csv(data_path)

# /////////////// Do procoessing here ///////////////
X_data, y_data, encoder, lb = process_data(
    data, categorical_features=cat_features, label="salary", training=True
)


# /////////////// Train model on different splits  ///////////////
print("------------------")
metrics_df, metrics_mean = train_and_test_on_slices(
    X_data, y_data, test_size_default=0.2
)
print("All training metrics gone through")
print(metrics_df)
print("Mean of all training metrics")
# format the metrics print to have only 2 decimal places
print(
    f"Precision: {metrics_mean['precision']:.2f}, Recall: {metrics_mean['recall']:.2f}, F1: {metrics_mean['f1']:.2f}, Accuracy: {metrics_mean['accuracy']:.2f}"
)
# find the row with the best f1 score
best_row = metrics_df.loc[metrics_df["f1"].idxmax()]
# print the best row
print("Best row")
# format the outputs to have only 2 decimal places
print(
    f"random_state: {best_row['random_state']}, test_size: {best_row['test_size']}, precision: {best_row['precision']:.2f}, recall: {best_row['recall']:.2f}, f1: {best_row['f1']:.2f}, accuracy: {best_row['accuracy']:.2f}"
)
# extract the random state and the test size of the best row
random_state = int(best_row["random_state"])
test_size = best_row["test_size"]


# ///////////// Create final model with best split ///////////////////////
# Optional enhancement, use K-fold cross validation instead of a train-test split.
print("------------------")
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=test_size, random_state=random_state
)
# Train and save a model.
model = train_model(X_train, y_train)
# evaulate model
# create a list of predictions from the model
print("------------------")
predictions = model.predict(X_test)
precision, recall, fbeta = compute_model_metrics(y_test, predictions)
print("Final model metrics")
print_metrics(precision, recall, fbeta, model.score(X_test, y_test))
final_metrics = pd.DataFrame(
    [[precision, recall, fbeta, model.score(X_test, y_test)]],
    columns=["precision", "recall", "f1", "accuracy"],
)

# ///////////// Store results in a file ///////////////////////
# create a pandas dataframe for final metrics


# save the model as a pkl file.
model_path = "starter/model/model.pkl"
encoder_path = "starter/model/encoder.pkl"
lb_path = "starter/model/lb.pkl"

pkl.dump(model, open(model_path, "wb"))
pkl.dump(encoder, open(encoder_path, "wb"))
pkl.dump(lb, open(lb_path, "wb"))
