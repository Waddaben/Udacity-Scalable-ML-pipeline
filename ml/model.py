# pylint: disable=too-many-locals
"""
This is the docstring for the model
"""
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from ml.data import process_data


# Optional: implement hyperparameter tuning.
def train_model(training_data, training_labels, random_state=42):
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
    model.fit(training_data, training_labels)
    return model


def compute_model_metrics(labels, preds):
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
    fbeta = fbeta_score(labels, preds, beta=1, zero_division=1)
    precision = precision_score(labels, preds, zero_division=1)
    recall = recall_score(labels, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, data):
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
    preds = model.predict(data)
    return preds


def train_and_test_on_slices(
    training_datasets, testing_datasets, test_size_default=0.2
):
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
        training_data, testing_data, training_label, testing_label = train_test_split(
            training_datasets,
            testing_datasets,
            test_size=test_size_default,
            random_state=i,
        )
        model = train_model(training_data, training_label)
        predictions = model.predict(testing_data)
        precision, recall, fbeta = compute_model_metrics(testing_label, predictions)
        print_metrics(
            precision, recall, fbeta, model.score(testing_data, testing_label)
        )
        metrics.append(
            [
                int(i),
                test_size_default,
                precision,
                recall,
                fbeta,
                model.score(testing_data, testing_label),
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


def evaluate_with_feature_fixed( # pylint: disable=too-many-arguments
    model, training_data, fixed_metric, cat_features, encoder, label_binarizer
):
    """
    This is a function that computes the performance metrics when
    the value of a given feature is held fixed

    Inputs
    ------
    model : ML model
        Trained machine learning model.
    training_data: pd.DataFrame
        The data to be used for evaluation
    fixed_metric : str
        The name of the feature to be held fixed
    encoder : sklearn.preprocessing.OneHotEncoder
        The encoder used to encode the categorical features
    label_binarizer :  sklearn.preprocessing.LabelBinarizer
        The label binarizer used to binarize the labels

    Returns
    -------
    None
    """
    # Get the unique values of the feature of interest
    unique_values = training_data[fixed_metric].unique()

    # Creating a txt file where we'll write our perfromance results
    with open(f"performance_{fixed_metric}.txt", "w", encoding="utf-8") as file:
        # Looping through the unique values of the feature of interest
        # Iterating over each slice and calculating the performance metrics
        # create a title for the file
        file.write(f"Performance metrics for {fixed_metric}")
        file.write("\n")
        file.write("-" * 10)
        file.write("\n")
        file.write("-" * 10)
        file.write("\n")
        for fixed_slice in unique_values:
            file.write(fixed_slice)
            file.write("\n")
            metric_fixed_df = training_data.loc[
                training_data.loc[:, fixed_metric] == fixed_slice, :
            ]
            # Process the test data with the process_data function.
            all_data_processed, all_labels_processed, encoder, label_binarizer = process_data(
                metric_fixed_df,
                categorical_features=cat_features,
                label="salary",
                training=False,
                encoder=encoder,
                label_bin=label_binarizer,
            )
            predictions = inference(model, all_data_processed)
            precision, recall, fbeta = compute_model_metrics(all_labels_processed, predictions)
            # Write the metrics to the file
            file.write(f"Precision: {precision}\n")
            file.write(f"Recall: {recall}\n")
            file.write(f"fbeta: {fbeta}\n")
            file.write(f"Accuracy: {model.score(all_data_processed, all_labels_processed)}\n")
            file.write("-" * 10)
            file.write("\n")
        file.close()


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
