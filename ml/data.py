# pylint: disable=too-many-arguments
"""
Module docstring for data
"""
import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data(
    dataset,
    categorical_features,
    label=None,
    training=True,
    encoder=None,
    label_bin=None,
):
    """Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        ydata = dataset[label]
        dataset = dataset.drop([label], axis=1)
    else:
        ydata = np.array([])

    xdata_categorical = dataset[categorical_features].values
    xdata_continuous = dataset.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        label_bin = LabelBinarizer()
        xdata_categorical = encoder.fit_transform(xdata_categorical)
        ydata = label_bin.fit_transform(ydata.values).ravel()
    else:
        xdata_categorical = encoder.transform(xdata_categorical)
        try:
            ydata = label_bin.transform(ydata.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    dataset = np.concatenate([xdata_continuous, xdata_categorical], axis=1)
    return dataset, ydata, encoder, label_bin
