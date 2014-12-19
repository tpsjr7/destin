import cv2
import numpy as np
import pydestin as pd
import random
import datetime


max_x = 512
max_y = 512
hidden_units = 10
cross_valid_size = 50
nn_save_file = "nn_save.xml"
input_video = "./finger.mov"

train_destin = False
load_destin = "train_nn.dst"
save_destin = "train_nn.dst"

#openvc python bindings don't expose CvTermCriteria, so we defined it in destin bindings

def makeParams():
    term_crit = pd.cvTermCriteria(pd.CV_TERMCRIT_ITER | pd.CV_TERMCRIT_EPS, 20000, 0.0000001)
    bp_dw_scale = 0.1
    bp_moment_scale = 0.1
    p = pd.CvANN_MLP_TrainParams(term_crit, int(cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP), bp_dw_scale, bp_moment_scale)
    print dir(p)
    pd.dumpParams(p)
    return p

def train():
    layers = np.array([2, 4, 1])
    nn = cv2.ANN_MLP(layers, cv2.ANN_MLP_SIGMOID_SYM, 1.0, 1.0)

    data = np.matrix("0 0; 0 1 ; 1 0 ; 1 1", np.float32)
    truth = np.matrix("0 ; 1 ; 1 ; 0", np.float32)


    nn.train(data, truth, None, None, makeParams())
    #nn.train(data, truth, None, np.array([0,1,2,3]))
    #nn.train(data, truth, None)

    dummy, outputs = nn.predict(data)
    print dummy, outputs
    print datetime.datetime.now()
    print data
    print truth
    print "Iterations: ", dummy

def loadTruthLabels():
    f = open("./finger_data.txt")
    frames = []
    xs = []
    ys = []

    for line in f:
        parts = line.split()
        frames.append(int(parts[0]))
        xs.append(float(parts[1]) / max_x)
        ys.append(float(parts[2]) / max_y)

    print "Loaded %d examples." % (len(frames))
    return (frames,xs,ys)

def createFeatures(dn, vs):
    be = pd.BeliefExporter(dn, 0)
    vs.rewind()
    frames, xs, ys = loadTruthLabels()

    # for training
    nn_train_features = []
    nn_train_lables = []

    # cross validation
    cv_features = []
    cv_labels = []

    # randomize the data
    ri = range(len(frames)) # = random inicies
    random.shuffle(ri)

    # helper function
    def runFrame(i):
        vs.setNextFrame(frames[ri[i]])
        vs.grab()
        for j in range(dn.getLayerCount()): # run enough times to flush out the data from prev frames
            dn.doDestin(vs.getOutput())

    # create train features
    for i in xrange(len(frames) - cross_valid_size):
        runFrame(i)
        nn_train_features.append(be.getBeliefsNumpy(be.getOutputSize()))
        nn_train_lables.append([xs[ri[i]], ys[ri[i]]])

    # create cross validation features
    for i in xrange(len(frames) - cross_valid_size + 1, len(frames)):
        runFrame(i)
        cv_features.append(be.getBeliefsNumpy(be.getOutputSize()))
        cv_labels.append([xs[ri[i]], ys[ri[i]]])

    return (np.matrix(nn_train_features),
             np.matrix(nn_train_lables),
             np.matrix(cv_features),
             np.matrix(cv_labels))

def trainDestin():
    vs = pd.VideoSource(False, input_video)
    width = 256
    vs.setSize(width,width)
    layers = 7
    centroids = [5,5,5,5,5,5,5]
    dn = pd.DestinNetworkAlt(width, layers, centroids, True)

    train_iterations = 600

    dn.setIsTraining(False)

    for stage in xrange(layers):
        if stage > 0:
            dn.setLayerIsTraining(stage - 1, False)
        dn.setLayerIsTraining(stage, True)

        for i in xrange(train_iterations):
            if vs.grab():
                dn.doDestin(vs.getOutput())
                if i % 10 == 0:
                    print "S: %d I: %d Q: %f" % (stage, i, dn.getQuality(stage))

    dn.setIsTraining(False)
    dn.save(save_destin)

    return (dn, vs)

def trainNN(features, labels):
    nn_layers = np.array([np.size(features,1), hidden_units, np.size(labels, 1)])
    nn = cv2.ANN_MLP(nn_layers, cv2.ANN_MLP_SIGMOID_SYM, 1.0, 1.0)

    print "Starting NN Training..."
    nn.train(features, labels, None, None, makeParams())
    print "Finished NN Training"
    #nn.train(data, truth, None, np.array([0,1,2,3]))
    #nn.train(data, truth, None)
    nn.save(nn_save_file)
    return nn

def meanSquaredError(actual, expected):
    return np.power((actual - expected),2).mean()

def checkAccuracy(neural_net, train_features, train_labels, cv_features, cv_labels):
    dummy, train_preds = neural_net.predict(train_features)
    dummy, cv_preds = neural_net.predict(cv_features)
    print "MSE Train:", meanSquaredError(train_preds, train_labels)
    print "MSE CV:", meanSquaredError(cv_preds, cv_labels)


def run():
    if train_destin:
        dn, vs = trainDestin()
    else:
        dn = pd.DestinNetworkAlt(load_destin)
        dn.setIsTraining(False)
        vs = pd.VideoSource(False, input_video)

    features, labels, cv_features, cv_labels = createFeatures(dn, vs)
    nn = trainNN(features, labels)

    checkAccuracy(nn, features, labels, cv_features, cv_labels)

run()
