import numpy as np
import cv2
from scipy import linalg
from sklearn.utils import array2d, as_float_array
from sklearn.base import TransformerMixin, BaseEstimator

def _visualizeEignvalues(zca, which=0):
    comp = zca.U_[:,which]
    width = np.sqrt(comp.shape[0])
    
    rescaled = (comp - comp.min()) / comp.ptp() * 255.0
    rescaled = rescaled.reshape(width, width).astype(np.uint8)
    img = cv2.resize(rescaled, (256,256))
    cv2.imshow("components", img)
    cv2.waitKey(100)   
    

class ZCA(BaseEstimator, TransformerMixin):

    def __init__(self, regularization=10**-5, copy=False):
        self.regularization = regularization
        self.copy = copy

    def fit(self, X, y=None):
        X = array2d(X)
        X = as_float_array(X, copy = self.copy)
        X -= X.mean(axis=1)[np.newaxis].T
        sigma = X.T.dot(X) / X.shape[0]
        U, S, V = linalg.svd(sigma)
        self.U_ = U
        self.S_ = S
        tmp = U.dot(np.diag(1/np.sqrt(S+self.regularization)))
        self.components_ = tmp.dot(U.T)

        return self

    def transform(self, X):
        X = array2d(X)
        t = X - X.mean(axis=1)[np.newaxis].T
        t = t.dot(self.components_.T)
        return t

    def visualizeEignvalues(self, which=0):
        _visualizeEignvalues(self, which)
        
    def varianceReport(self, fraction=0.99):
        size = self.S_.shape[0]
        r = np.bincount(np.cumsum(self.S_) / self.S_.sum() > fraction)[0] / float(size)
        print "Need to keep",r,"of the features, or",int(np.ceil(r*size)),"of",size
        
        
