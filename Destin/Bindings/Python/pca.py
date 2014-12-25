import numpy as np
import pydestin as pd
import cv2

x = np.load("finger_features.npy")[0]
m = x.shape[0]
n = x.shape[1]

def norm(x):
    mu = x.mean(0)
    x_norm = x - mu
    sigma = x_norm.std(0)
    x_norm = x_norm / sigma
    return (x_norm, mu, sigma)

x_norm, mu, sigma = norm(x)

cov = 1.0  / m * x_norm.T.dot(x_norm)
u,s,v = np.linalg.svd(cov)

var_retained = .95

def calc_frac_needed(s, var_retained):
    stat = np.cumsum(s) / sum(s)
    num_needed_features = np.bincount(stat >= var_retained)[0]
    frac_needed = num_needed_features / float(n)
    print "Need to keep %d (%f) of the features." % (num_needed_features, frac_needed)
    return frac_needed

calc_frac_needed(s, var_retained)
