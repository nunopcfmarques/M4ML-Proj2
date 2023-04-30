
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

SEED = 95758

def ffs(X, target, OF, **kwargs):
    '''
    Input:  X - pandas dataframe with the training data
            target - pandas dataframe with the training target
            OF - objective function to use
            **kwargs - additional arguments for the objective function
    
    Output: S - list with the names of the selected features

    Description: Selects the k best features using a forward feature selection algorithm
    '''

    # Initialize the list of selected features to empty
    kwargs["S"] = []
    # Initialize the list of remaining features to all features
    F = list(X.columns.values)

    for k in range(len(F)):
        # Initialize the dictionary of objective function values for each feature
        values = {}
        for feature in F:
        # Compute the objective function value for each feature
            values[feature] = OF(X, feature, target, **kwargs)
            
        # Select the feature with the highest objective function value
        X_j = max(values, key=values.get)
        # Add the selected feature to the list of selected features
        kwargs["S"].append(X_j)
        # Remove the selected feature from the list of remaining features
        F.remove(X_j)

    return kwargs["S"]

def MIM(X, feature, train_target, **kwargs):
    '''
    Input:  train_df - pandas dataframe with the training data
            train_target - pandas dataframe with the training target
            k - number of features to select
    
    Output: filter - list with the names of the selected features

    Description: Selects the k best features using SelectKBest and mutual information
    '''

    return mutual_info_classif(X[feature].to_frame(), train_target, random_state=SEED)

def MIFS(X, feature, target, **kwargs):
    '''
    Input:  X - pandas dataframe with the training data
            feature - name of the feature to compute the objective function value
            target - pandas dataframe with the training target
            **kwargs - additional arguments for the objective function
    
    Output: objective function value

    Description: Computes the objective function value for a feature using the MIFS algorithm
    '''
    sum = 0
    for selected_feature in kwargs["S"]:
        sum = sum + mutual_info_regression(X[selected_feature].to_frame(), X[feature], random_state=SEED)
    return mutual_info_classif(X[feature].to_frame(), target, random_state=SEED) - (kwargs["beta"] * sum)

def mRMR(X, feature, target, **kwargs):
    '''
    Input:  X - pandas dataframe with the training data
            feature - name of the feature to compute the objective function value
            target - pandas dataframe with the training target
            **kwargs - additional arguments for the objective function

    Output: objective function value

    Description: Computes the objective function value for a feature using the mRMR algorithm
    '''
    sum = 0
    for selected_feature in kwargs["S"]:
        sum = sum + mutual_info_regression(X[selected_feature].to_frame(), X[feature], random_state=SEED)
    if (np.size(kwargs["S"]) == 0):
        return mutual_info_classif(X[feature].to_frame(), target, random_state=SEED)
    else:
        return mutual_info_classif(X[feature].to_frame(), target, random_state=SEED) - ((1 / np.size(kwargs["S"])) * sum)
    
def maxMIFS(X, feature, target, **kwargs):
    '''
    Input:  X - pandas dataframe with the training data
            feature - name of the feature to compute the objective function value
            target - pandas dataframe with the training target
            **kwargs - additional arguments for the objective function
    
    Output: objective function value

    Description: Computes the objective function value for a feature using the maxMIFS algorithm
    '''
    if (np.size(kwargs["S"]) == 0):
        return mutual_info_classif(X[feature].to_frame(), target, random_state=SEED)
    else:
        selected_features_values = [mutual_info_regression(X[selected_feature].to_frame(), X[feature], random_state=SEED) for selected_feature in kwargs["S"]]
        return mutual_info_classif(X[feature].to_frame(), target, random_state=SEED) - (np.max(selected_features_values))
    

def get_metrics(test_target, predict):
    '''
    Input:  test_target - pandas dataframe with the test target
            predict - pandas dataframe with the predicted target

    Output: acc - accuracy score, rec - macro recall score, pre - macro precision score, f1 - macro f1 score
    '''
    acc = accuracy_score(test_target, predict)
    rec = recall_score(test_target, predict, average='macro')
    pre = precision_score(test_target, predict, average='macro')
    f1 = f1_score(test_target, predict, average='macro')

    return acc, rec, pre, f1
