import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, SelectKBest
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def MIM(train_df, train_target, k):
    selector = SelectKBest(mutual_info_classif, k=k)
    X_reduced = selector.fit_transform(train_df, train_target)

    filter = selector.get_feature_names_out()
    return filter

def ffs(X, target, OF, k, **kwargs):
    S = []
    F = list(X.columns.values)

    for i in range(k):
        values = {}

        for feature in F:
            values[feature] = OF(X, feature, target, S, **kwargs)
        X_j = max(values, key=values.get)
        S.append(X_j)
        F.remove(X_j)

    return S

def MIFS(X, feature, target, S, **kwargs):
    sum = 0
    for selected_feature in S:
        sum = sum + mutual_info_regression(X[selected_feature].to_frame(), X[feature])
    return mutual_info_classif(X[feature].to_frame(), target) - (kwargs["beta"] * sum)

def mRMR(X, feature, target, S, **kwargs):
    sum = 0
    for selected_feature in S:
        sum = sum + mutual_info_regression(X[selected_feature].to_frame(), X[feature])
    if (np.size(S) == 0):
        return mutual_info_classif(X[feature].to_frame(), target)
    else:
        return mutual_info_classif(X[feature].to_frame(), target) - ((1 / np.size(S)) * sum)
    
def maxMIFS(X, feature, target, S, **kwargs):
    if (np.size(S) == 0):
        return mutual_info_classif(X[feature].to_frame(), target)
    else:
        selected_features_values = [mutual_info_regression(X[selected_feature].to_frame(), X[feature]) for selected_feature in S]
        return mutual_info_classif(X[feature].to_frame(), target) - (np.max(selected_features_values))
    

def get_metrics(test_target, predict):
    acc = accuracy_score(test_target, predict)
    rec = recall_score(test_target, predict, average='macro')
    pre = precision_score(test_target, predict, average='macro')
    f1 = f1_score(test_target, predict, average='macro')

    return acc, rec, pre, f1
