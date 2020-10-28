import numpy as np
import logging
import pickle
import time
import sklearn

__version__ = "0.1"
logger = logging.gerLogger(__name__)

class NBmodel(object):
    def __init__(self):
        logger.info('Starting %s Microservice, version %s', __name__, __version__)
        self.model = pickle.load(open("NBmodel.pkl", "rb"))

    def predict(self, X, features_names=None):
        print(X)
        return self.model.predict([X])
