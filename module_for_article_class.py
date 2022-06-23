# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 15:31:23 2022

@author: ANAS
"""
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
        

class ModelCreation():
    
    def __init__(self):
        pass

    def model_evaluation(self, model,y_test, X_test):
        '''
        

        Parameters
        ----------
        y_true : SERIES
            DESCRIPTION.
        y_pred : SERIES
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        y_true = y_test
        y_pred = model.predict(X_test)

        y_true = np.argmax(y_true,axis=1)
        y_pred = np.argmax(y_pred,axis=1)

        print(classification_report(y_true, y_pred))
        print(accuracy_score(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))