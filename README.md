# Udacity-Arvato-Project

## Project Overview
Arvato Financial Solutions is a Germany-based company providing B2B financial Service. Utilizing the dataset from Arvato, the objectives of this project are as follows:

> 1.   Help the company identify the customer segment by unsupervised learning;
> 2.   Improve marketing efficiency by supervised learning models.

## Problem Statement

### Customer segment report
 - Question: Which customer attributes are significantly differentiating from the non-customer population?
 - Metrics: **Percent of Customers in the group.** When a group in clustering result has a high percentage of customers, the attribute of this group can represent the customer segment of this company.
 - Strategy: Unsupervised learning(PCA and K-Means clustering)
 - Expected Solution: 

> 1.   Divide the population into several groups.
> 2.   Find the group which is most representative of the customer.
> 3.   Conclude the common characters of this group. 

### Marketing prediction analyze 
 - Question: Will a given candidate accept the company's offer through the marketing campaign?
 - Metrics:   Accuracy will assign an equal cost to false positives and false negatives. If the model predicts all candidates as false, the accuracy will still high if the dataset is highly imbalanced. Therefore, we need to separate false positives and false negatives, which means use precision, Recall, and AUC to show the ability of the classifier to distinguish between classes.
  
> 1.   **Precision**: the fraction of relevant instances among the retrieved instances.
> 2.   **Recall**: the fraction of the total amount of relevant instances that were actually retrieved. 
> 3.   **AUC**: the ability of a classifier to distinguish between classes when the number is greater than 50%, which means the model is better than a random classifier.

 - Strategy: Supervised Machine Learning
 - Expected Solution: Classifier with high AUC score.


## Files:
> - data_wraggeling.ipynb: understand data and data preprocessing
> - customer_segment.ipynb: use PCA and K-means to perform customer segmentation
> - response_predict: build pipeline to predict marketing result 
> - src/*: pipeline scripe and model result dataset

## Libraries 
`numpy`, `pandas`, `sklearn`, `imblearn`, `pickle`
    
## Acknowledge
The dataset in the project prvided by Bertelsmann/Arvato. All the material is for Udacity Capstone project.