# Gaussian Naive Bayes Classifier for Predicting Basketball Game Outcomes

## Abstract
This project develops a Gaussian Naive Bayes Classifier aimed at predicting whether the home basketball team will win against the away team based on their recent game statistics. The report details the code architecture, data preprocessing, model building, results, and challenges encountered during development.

## 1. Introduction
The motivation for this project stems from the desire to statistically analyze basketball game outcomes using a probabilistic approach. The Gaussian Naive Bayes model offers a suitable framework due to its simplicity and efficacy in handling normally distributed data.

## 2. Code Architecture
The classifier is structured around several key and helper functions:
- `count_w`: Counts occurrences of wins in recent games.
- `preprocess`: Prepares the dataset for training and testing.
- `calc_mean_and_var`: Calculates the mean and variance for each feature.
- `gaussian_likelihood`: Computes the Gaussian likelihood for given statistics.
- `predict`: Predicts game outcomes based on computed likelihoods.
- `NaiveBayesClassifier`: Orchestrates preprocessing, training, and prediction processes.

## 3. Preprocessing
Data is first read from CSV files into a Pandas DataFrame. Features like the last five games' win-loss records are transformed into numerical values indicating the number of wins. Irrelevant or non-numerical features are dropped to focus on statistically significant predictors.

## 4. Model Building
The model separates training data by outcome (win/loss), computes statistics for each class, and uses these to calculate prior probabilities and likelihoods during prediction.

## 5. Results
The classifier achieves an accuracy of 66.9% on a test dataset, leaving a good bit of room for improvement. 

## 6. Challenges
Feature selection proved challenging, requiring iterative testing and domain knowledge to identify impactful predictors. Adjustments to feature engineering significantly influenced model performance, underscoring the importance of thoughtful statistical analysis.

## 7. Conclusion
While the Gaussian Naive Bayes Classifier shows promise in predicting basketball game outcomes, future work should explore more complex models and additional feature engineering techniques to enhance accuracy.

## Appendices
### A. Installation and Execution
- Required Python Version: 3.x
- Dependencies: pandas
- To run the model: `python NaiveBayesClassifier.py [train_data_path] [test_data_path]`

