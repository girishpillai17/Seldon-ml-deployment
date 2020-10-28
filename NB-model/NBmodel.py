import numpy as np
import logging
import pickle
import time
import sklearn

class NBModel(object):
    def __init__(self):
        self.model = pickle.load(open("NBmodel.pkl", "rb"))

    def predict(self, X, features_names=None):
        print(X)
        return self.model.predict([X])
