# Breast-Cancer-Detection
INTRODUCTION
It is a classification problem in machine learning. Data is provided having 10 features. I have used 5 different classsification algorithms and compared their predictions. I have used Jupyter framework while working on this project. I have used sklearn, numpy, pandas, matplotlb libraries to assist me in visualizing my results.

DATASET

The data has been got from the  following url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
The dataset used in this story is publicly available and was created by Dr. William H. Wolberg, physician at the University Of Wisconsin Hospital at Madison, Wisconsin, USA.

It has 10 features:  
'clump_thickness' :  
'uniform_cell_size', 
'uniform_cell_shape',
'marginal_adhesion',
'single_epithelial_size',
'bare_nuclei',
'bland_chromatin',
'normal_nucleoli',
'mitoses',
'class'

Here the class variable is the target variable. Our objective is to predict whether the given sample malignant or benign. It is a binary classification problem. 

Missing values are being ignored by replacing the " ? " with "-99999",python will ignore these values

Using Scatter matrix, we find that there is linear relationship between uniform_cell_size and uniform_cell_shape. Apart from that, for every values of independent variables, class variable is taking all the values so not a particular independent variable is deciding the class variable. 

For training and validation, dataset is splitted into 80/20 ratio. 

Five different algorithms have been used to classifyabove problem. 
1. Logistic regression
2. Support vector machine 
3. Naive bayes 
4. KNN: used 5 as number of neighbours 
5. Random Forest 

TRAINING 

Here is the training accuracy for these models
KNN: 0.957045 
SVM: 0.647597 
LogReg: 0.930195 
Naive_Bayes: 0.948117
Random_forest: 0.958864 

We see that SVM dosent fit well while random forest and KNN fits well

TESTING

KNN
0.9857142857142858
              precision    recall  f1-score   support

           2       0.99      0.99      0.99        96
           4       0.98      0.98      0.98        44

    accuracy                           0.99       140
   macro avg       0.98      0.98      0.98       140
weighted avg       0.99      0.99      0.99       140

SVM
0.6857142857142857
              precision    recall  f1-score   support

           2       0.69      1.00      0.81        96
           4       0.00      0.00      0.00        44

    accuracy                           0.69       140
   macro avg       0.34      0.50      0.41       140
weighted avg       0.47      0.69      0.56       140

LogReg
0.9857142857142858
              precision    recall  f1-score   support

           2       1.00      0.98      0.99        96
           4       0.96      1.00      0.98        44

    accuracy                           0.99       140
   macro avg       0.98      0.99      0.98       140
weighted avg       0.99      0.99      0.99       140

Naive_Bayes
0.9642857142857143
              precision    recall  f1-score   support

           2       0.97      0.98      0.97        96
           4       0.95      0.93      0.94        44

    accuracy                           0.96       140
   macro avg       0.96      0.96      0.96       140
weighted avg       0.96      0.96      0.96       140

Random_forest
0.9785714285714285
              precision    recall  f1-score   support

           2       0.98      0.99      0.98        96
           4       0.98      0.95      0.97        44

    accuracy                           0.98       140
   macro avg       0.98      0.97      0.97       140
weighted avg       0.98      0.98      0.98       140

 We see that KNN and logistic regression performs really well having F1score of 0.99. 
 PS: F1 score is 2 (Precision * Recall)/ (Precision + recall). It is better as it takes into account both recall and precison values.
