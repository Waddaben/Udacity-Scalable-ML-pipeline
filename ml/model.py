import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train, random_state=42):
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
    # use X_train and y_train to train a random forest classifier.
    model = RandomForestClassifier(random_state)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
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


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : model object
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def train_and_test_on_slices(training_datasets, testing_datasets, test_size_default=0.2):
    """
    Write a function that outputs the performance of the model on slices of the data
    Args:
        training_datasets (pd.DataFrame): The features of the data.
        testing_datasets (pd.Series): The labels of the data.
    Returns:
    None
    """
    # create an array to store the metrics
    metrics = []

    # split the data into 10 slices and do k fold cross validation on each slice
    for i in range(10):
        # print that we are on the ith slice
        print("------------------")
        print(f"Slice {i}:")
        X_train, X_test, y_train, y_test = train_test_split(
            training_datasets, testing_datasets, test_size=test_size_default, random_state=i
        )
        model = train_model(X_train, y_train)
        predictions = model.predict(X_test)
        precision, recall, fbeta = compute_model_metrics(y_test, predictions)
        print_metrics(precision, recall, fbeta, model.score(X_test, y_test))
        metrics.append(
            [
                int(i),
                test_size_default,
                precision,
                recall,
                fbeta,
                model.score(X_test, y_test),
            ]
        )
    # create a pandase dataframe from the metrics array
    metrics_df = pd.DataFrame(
        metrics,
        columns=["random_state", "test_size", "precision", "recall", "f1", "accuracy"],
    )
    # create a pandase dataframe with the mean of the metrics
    metrics_mean = metrics_df.mean()
    return metrics_df, metrics_mean


def print_metrics(precision, recall, fbeta, accuracy):
    """
    Print the metrics of the model
    Args:
        precision (float): The precision of the model
        recall (float): The recall of the model
        fbeta (float): The fbeta of the model
        accuracy (float): The accuracy of the model
    Returns:
    None
    """
    # print the metrocs but round them to 2 decimal places
    print(f"Precision: {round(precision, 2)}")
    print(f"Recall: {round(recall, 2)}")
    print(f"F1: {round(fbeta, 2)}")
    print(f"Accuracy: {round(accuracy, 2)}")
