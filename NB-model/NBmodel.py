import numpy as np
import logging
import pickle
import time
import sklearn

class NBModel(object):
    def __init__(self):
        self.model = pickle.load(open("NBmodel.pkl", "rb"))
    
    """
    wrap your machine learning model create a Class that has a predict method
    Return a prediction.

    Parameters
    ----------
    X : array-like
    feature_names : array of feature names (optional)
    """
    def predict(self, X, features_names=None):
        print(X)
        return self.model.predict([X])
