#!/usr/bin/env python3

import os
import pandas as pd
# import numpy as np
from sklearn import (svm, preprocessing)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (recall_score, precision_score, accuracy_score, confusion_matrix,)
#precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# load and preprocess data
DATA = "/home/reddy/Documents/AI_ML_DL/2_Kaggle/breast_cancer/data.csv"

def liver_data():
    data = pd.read_csv(DATA)
    # data.info()
    data.head()
    data.tail()
    data.describe()
    # data_bk = data.copy()
    # data_nan = data[data.isna().any(axis=1)] # rows having NaN
    # nan_rows = list(data_nan.index)
    # data_types = data.dtypes
    # print(data_types)
    # print("Rows having NaN/missing values are {}".format(nan_rows))
    # data.groupby("diagnosis").size()
    # data.groupby("Gender").size()
    # features = data.columns.tolist()
    # data_nan = data[data.isna().any(axis=1)]

    rm_cols = (0, 32)
    data.drop([data.columns[i] for i in rm_cols], axis=1, inplace=True)
    data.dropna(axis=0, how='any', inplace=True)
    data["diagnosis"].value_counts()
    mldata = data.drop("diagnosis", axis=1)
    labels = data["diagnosis"].map({"M":1, "B":0})

    label_ecoder = LabelEncoder()
    label_ecoder.fit(labels)
    labels = label_ecoder.transform(labels)    # apply encoding to labels

    # classifiers
    svm_classifier = svm.SVC()
    rf_classifier = RandomForestClassifier(random_state=0)
    logres_classifier = LogisticRegression(multi_class='multinomial', solver='newton-cg')

    print("{} results are:-\n".format("SVM"))
    classify_data(svm_classifier, mldata, labels)

    print("\n{} results are:-\n".format("Random Forest"))
    classify_data(rf_classifier, mldata, labels)

    print("\n{} results are:-\n".format("Logistic Regression"))
    classify_data(logres_classifier, mldata, labels)


def classify_data(classifier, mldata, labels):
    #preprocessing data using sk.learn
    data_variables = train_test_split(mldata, labels, test_size=0.2, random_state=970)
    train_data, test_data, train_label, test_label = data_variables
    scaler = preprocessing.StandardScaler().fit(train_data)
    train_data_scaled = scaler.transform(train_data)
    test_data_scaled = scaler.transform(test_data)


    # classifier = RandomForestClassifier(random_state=0)
    classifier.fit(train_data_scaled, train_label)
    predict_y = classifier.predict(test_data_scaled)
    accuracy = classifier.score(test_data_scaled, test_label)
    accuracy = accuracy_score(test_label, predict_y)
    precision = precision_score(test_label, predict_y)
    recall = recall_score(test_label, predict_y)
    cmatrix = confusion_matrix(test_label, predict_y)
    print("accuracy  :- {}".format(accuracy))
    print("precision :- {}".format(precision))
    print("recall    :- {} \n".format(recall))
    # print("\nConfusion matrix \n{}\n\n".format(cmatrix))


if __name__ == '__main__':
    os.system("clear||cls")
    liver_data()
