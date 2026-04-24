import random
import numpy as np 

class OU(object):

    def function(self, x, mu, theta, sigma):
        # Return a scalar noise sample; Keras/TensorFlow callers expect float.
        return float(theta * (mu - x) + sigma * np.random.randn())