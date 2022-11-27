import pandas  as pd
import scipy
import numpy as np
import re
import json

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

from sklearn.metrics import balanced_accuracy_score,accuracy_score,recall_score,roc_auc_score,f1_score,precision_score,plot_confusion_matrix

import matplotlib.pyplot as plt
#================================ GET X AND Y ==================================#
def data_trasnform(df: pd.DataFrame,smote = False):
    '''
        this function apply encoding, standar scaler , split 
        independent variabe and dependet variable and SMOTE
        
        inputs:
            df: pd.DataFrame
            smote: default False
    '''
    
    # ============== Feature engeniering======================#
    df['age_group'] =  pd.cut(x=df['age'], bins=[15,30,40,50,60,100],\
                    labels= ['15-30' , '30 -40' , '40 - 50', '50 - 60' , '60 - mas'])
    
    #===============One Hot Encoding=======================#
    df=pd.get_dummies(data=df, drop_first=True)
    
    # ============== STANDARD SCALER=======================#
    scaler = StandardScaler()
    df_scaler = scaler.fit_transform(df)
    df_scaler = pd.DataFrame(df_scaler, columns=df.columns)
    
    def change_target(x):
        '''
        Transforma la variable target en 0 y 1 luego del Scaler
        '''
        if x < 0:
            return 0
        else:
            return 1
    
    df_scaler['y_yes'] = df_scaler['y_yes'].apply(change_target)
    
    # =====split dependent and independent feature===========# 
    X = df_scaler.loc[:,df.columns != 'y_yes'].values
    y = df_scaler.loc[:,'y_yes'].values
    
    #=========== Synthetic Minority Oversamping Technique=============#
    if smote:
        
        #CLASS RATIO TARGET DATASET
        original_ = sum(df['y_yes'])/len(df['y_yes'])
        #print('Class Ratio 1 ORIGINAL',original_)
        
        sm = SMOTE()
        X, y = sm.fit_resample(X, y)
        print('X.shape:',X.shape,'y.shape:',y.shape)
        
    return X, y

def plot_matrix(clf,X,y):
    name_model = str(type(clf))
    regex_model = "'(.*)'"
    color = 'black'
    name_model = re.search(regex_model,name_model).group(1)
    matrix = plot_confusion_matrix(clf, X, y, cmap=plt.cm.Blues)
    matrix.ax_.set_title('Confusion Matrix - {}'.format(name_model), color=color)
    plt.xlabel('Predicted Label', color=color)
    plt.ylabel('True Label', color=color)
    return plt.show()

def classifier_SKF(clf,X,y, n_splits = 10):
    
    skf = StratifiedKFold(n_splits=n_splits)

    balanced_accuracy_score_list = []
    accuracy_score_list = []
    roc_auc_score_list = []
    f1_score_list = []
    precision_score_list = []
    recall_score_list = []

    for idx in skf.split(X, y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = X[train_idx]
        ytrain = y[train_idx]

        xtest = X[test_idx]
        ytest = y[test_idx]

        clf.fit(xtrain,ytrain)
        preds = clf.predict(xtest)

        #===============balance_accuracy_score=========#
        fold_acc = balanced_accuracy_score(ytest, preds)
        balanced_accuracy_score_list.append(fold_acc)

        #==============Accuracy_score==================#
        fold_acc = accuracy_score(ytest, preds)
        accuracy_score_list.append(fold_acc)

        #=================roc_auc_score================#
        fold_acc = roc_auc_score(ytest, preds,multi_class='ovr')
        roc_auc_score_list.append(fold_acc)

        #==================f1_score====================#
        fold_acc = f1_score(ytest, preds)
        f1_score_list.append(fold_acc)

        #==============precision_score=================#
        fold_acc = precision_score(ytest, preds)
        precision_score_list.append(fold_acc)

        #================recall_score==================#
        fold_acc = recall_score(ytest, preds)
        recall_score_list.append(fold_acc)


    # ======================= Metrics======================================#    
    balance_accuracy_score_ = scipy.stats.gmean(balanced_accuracy_score_list)

    accuracy_score_ = scipy.stats.gmean(accuracy_score_list)

    roc_auc_score_ = scipy.stats.gmean(roc_auc_score_list)

    f1_score_ = scipy.stats.gmean(f1_score_list)

    precision_score_ = scipy.stats.gmean(precision_score_list)

    recall_score_ = scipy.stats.gmean(recall_score_list)

    data = [['balance_accuracy_score',balance_accuracy_score_],
            ['accuracy_score',accuracy_score_],
            ['roc_auc_score',roc_auc_score_],
            ['f1_score',f1_score_],
            ['precision_score',precision_score_],
            ['recall_score',recall_score_]]
    columns = ['Metrics','Values']
    df = pd.DataFrame(data = data, columns=columns)
    display(df)
    
    #print confusion matrix
    plot_matrix(clf,X,y)   
    return clf

def read_hyperameter_json(path = '' ):
    with open(path, 'r') as openfile:
        values = json.load(openfile)
    print('Hyperameter:  ',values)
    return values

def save_hyperameter_json(dictionary,path = '' ):
    with open(path, "w") as outfile:
        json.dump(dictionary, outfile)
        print('Save ready')