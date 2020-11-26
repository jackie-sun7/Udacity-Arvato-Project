# Udacity-Arvato-Project

## Project Overview
Arvato Financial Solutions is a Germany-based company providing B2B financial Service. Utilizing the dataset from Arvato, the objectives of this project are as follows:

> 1.   Help the company identify the customer segment by unsupervised learning;
> 2.   Improve marketing efficiency by supervised learning models.

### Motivation
Mailing is one of the most comment marketing strategy. The marketing department sends out hundreds of mails every single day, but every mail has its cost. Therefore, mailing to the right people will increase the response rate and help the company decrease the marking expense. To find the 'right people', we explore the dataset from Arvato Financial Solutions to answer the two questions below.

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
 
## Result Summary

In the project, we first explore and preprocess the dataset in four steps:

 - 1.Conver mix type to int
 - 2.Fill missing value
 - 3.Handle outliers
 - 4.rop columns lacking variance

Then, we analyze the dataset from two perspectives below:

> 1.   Help the company identify the customer segment by unsupervised learning;
> 2.   Improve marketing efficiency by supervised learning models.

In the first part, creating customer segments by unsupervised learning, we use PCA to reduce the dataset to 100 variables then divide the population into 17 groups. To research the attributes of the group with high customer rate, we find that **our target customers are the people who have frequent transaction activities**. 

In the second part, building a predictive model by supervised learning, we use SMOT to handle imbalanced datasets, select XGBoost from 3 candidate models and then do hyper-parameter tuning from two perspectives to handle overfitting: decrease the deep of the tree and subsampling. The model evaluation use in-sample and off-sample test with ROC-AUC score. 


## Programs:
> - data_wraggeling.ipynb: understand data and data preprocessing
> - customer_segment.ipynb: use PCA and K-means to perform customer segmentation
> - response_predict: build pipeline to predict marketing result 
> - src/*: pipeline scripe and model result dataset

## Libraries 
`numpy`, `pandas`, `sklearn`, `imblearn`, `pickle`
    
## Acknowledge
The dataset in the project prvided by Bertelsmann/Arvato. All materials are for Udacity Capstone project.