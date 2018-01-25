#!/usr/bin/env python3

import pandas as pd
# import numpy as np
# from sklearn import (svm, preprocessing)
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


# load and preprocess data
RAW_DATA = "/home/reddy/Documents/AI_ML_DL/2_Kaggle/Liver_disease/indian_liver_patient.csv"
SVM_DATA = "/home/reddy/Documents/AI_ML_DL/2_Kaggle/Liver_disease/py_data"

def load_data():
    data = pd.read_csv(RAW_DATA)
    data = data.dropna(how='any').replace("Male", 1).replace("Female", 0)
    gender = data.iloc[:, 1]
    labels = data.iloc[:, -1].replace(2, 0).values
    mldata = data.iloc[:, :-1].values

    # le = LabelEncoder().fit(gender)
    # gender = le.transform(gender)
    features = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
                'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens',
                'Albumin', 'Albumin_and_Globulin_Ratio', 'Dataset']

    plt.style.use('ggplot') # make plots look better
    data.plot.scatter(x="Total_Bilirubin", y="Total_Protiens")
    plt.show()

    sns.FacetGrid(data, hue="Dataset").map(plt.scatter, "Total_Bilirubin",
                                           "Total_Protiens").add_legend()

    plt.show()



if __name__ == '__main__':
    load_data()
