# Model Card

## Model Details

- The model was developed for the Udacity ML DevOps Nanodegree
- Created on 5th of October 2022
- Version 1.0
- Random forest classifier from sklearn with a random state of 42 and default hyperparamaters
- From <https://medium.com/analytics-vidhya/adult-census-income-dataset-using-multiple-machine-learning-models-f289c960005d> it was decided to use a random forest classifier
- Model was trained and evaluated on 10 different splits of which the split yielding the best model was used for the final model.

## Intended Use

- In this project, we will demonstrate how to deploy a scalable machine learning pipeline in production
- Based on census data, estimate whether income exceeds $50K/year
- Track the training of the data for reproducaibilty

## Training Data

- 80% of the data was used for training using from sklearn.model_selection import train_test_split
- with the best random state being 3

## Evaluation Data

- 20% of the data was used for evaluation with 10 different splits with different random states having been made. The average results of all these metrics were Precision: 0.73, Recall: 0.63, F1: 0.67, Accuracy: 0.86.

## Metrics

Metrics used and model's final results:
Precision: 0.75
Recall: 0.65
F1: 0.69
Accuracy: 0.86

## Ethical Considerations

The data should remain anonomyzed for the security of the locations and poeple living there.

## Caveats and Recommendations

The data was collected in 1994 and is therefore not that relevant anymore. Therefore it would be recommended to use newer data or adjust the old data accordingly.
