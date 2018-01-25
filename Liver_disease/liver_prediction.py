#!/usr/bin/env python3

import pandas as pd
# import numpy as np
from sklearn import (svm, preprocessing)
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (recall_score, precision_score, accuracy_score, confusion_matrix,)
#precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# load and preprocess data
DATA = "/home/reddy/Documents/AI_ML_DL/2_Kaggle/Liver_disease/indian_liver_patient.csv"


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
    # data.groupby("Dataset").size()
    # data.groupby("Gender").size()
    # max_index = data.iloc[:, 2:-1].idxmax(skipna=True)
    # data = data.dropna(axis=0, how='any', inplace=True).replace("Male", 1).replace("Female", 0)
    # features = data.columns.tolist()
    features = ['Age',
                'Gender',
                'Total_Bilirubin',
                'Direct_Bilirubin',
                'Alkaline_Phosphotase',
                'Alamine_Aminotransferase',
                'Aspartate_Aminotransferase',
                'Total_Protiens',
                'Albumin',
                'Albumin_and_Globulin_Ratio',]

    # data.drop([features[i] for i in [3, 5, 8]], axis=1, inplace=True)

    data["Gender"] = data["Gender"].map({"Male":1, "Female":0})
    data.dropna(axis=0, how='any', inplace=True)
    # data.fillna(data['Albumin_and_Globulin_Ratio'].mean(), inplace=True)
    data["Dataset"].value_counts()
    mldata = data.drop("Dataset", axis=1)
    labels = data["Dataset"].map({1:1, 2:0})
    # mldata = data.iloc[:, :-1].values
    # labels = data.iloc[:, -1].replace(2, 0).values
    # mldata = data.iloc[:, [2, 3, 4, 5, 6, 7]].values
    # mldata = data.iloc[:, [0, 2, 3, 4, 5, 6, 7, 8, 9]].values   #removed gender

    # classifier = svm.SVC()
    # classifier = RandomForestClassifier(random_state=0)
    classifier = LogisticRegression(multi_class='multinomial', solver='newton-cg')
    classify_data(classifier, mldata, labels)


def classify_data(classifier, mldata, labels):
    #preprocessing data using sk.learn
    data_variables = train_test_split(mldata, labels, test_size=0.2, random_state=970)
    train_data, test_data, train_label, test_label = data_variables
    scaler = preprocessing.StandardScaler().fit(train_data)
    train_data_scaled = scaler.transform(train_data)
    test_data_scaled = scaler.transform(test_data)

    # SVM classifier
    # classifier = svm.SVC(random_state=0)
    # classifier.fit(train_data, train_label)
    # predict_y = classifier.predict(test_data)
    # acc_test = classifier.score(test_data, test_label)
    # print(acc_test)

    # Random Forest classifier
    # classifier = RandomForestClassifier(min_samples_split=4)
    # classifier = RandomForestClassifier(min_samples_split=4, criterion="entropy")

    # classifier = RandomForestClassifier(random_state=0)
    classifier.fit(train_data_scaled, train_label)
    predict_y = classifier.predict(test_data_scaled)
    accuracy = classifier.score(test_data_scaled, test_label)
    # accuracy = accuracy_score(test_label, predict_y)
    precision = precision_score(test_label, predict_y)
    recall = recall_score(test_label, predict_y)
    cmatrix = confusion_matrix(test_label, predict_y)
    print(accuracy)
    print(precision)
    print(recall)
    print(cmatrix)



if __name__ == '__main__':
    liver_data()
