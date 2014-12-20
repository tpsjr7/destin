#!/usr/bin/python
import cv2
import numpy as np
import pydestin as pd
import random
import datetime

#1  2 4  8  16 32  64
#4  8 16 32 64 128 256
centroids = [5,5,5,5,5,5]
layers = len(centroids)

#when training samples were created, this was the video dimensions
labels_max_x = 512.0
labels_max_y = 512.0

# Size of the visualization window
visual_width_x = 512.0
visual_height_y = 512.0

# Size of the video fed to DeSTIN
destin_width = 4 * 2**(layers - 1)

hidden_units = 10

# number of samples used for cross validation
cross_valid_frac = 0.3

nn_save_file = "nn_save.xml"

position_train_data="./finger_data.txt"
input_video = "./finger.mov"

destin_train_iterations = 900

train_destin = False
train_nn = True

load_destin = "train_nn.dst"
save_destin = "train_nn.dst"

report_interval = 20

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
    print outputs

    print datetime.datetime.now()
    print data
    print truth
    print "Iterations: ", dummy

def scale_pos(fpos, min_pos, range_pos):
    """
    fpos - maxtrix of positions
    min_pos - array of minimums of each column of fpos
    range - array of ranges of each column of fpos
    """
    return (fpos - min_pos) / range_pos - 0.5

def unscale_pos(scaled_pos, min_pos, range_pos):
    return (scaled_pos + 0.5) * range_pos + min_pos

def loadTruthLabels():
    data = np.genfromtxt(position_train_data, dtype=np.int, invalid_raise=True)

    # frames used from the video
    frames = data[:,0].astype(np.int32)
    print "Loaded %d examples." % (frames.size)

    # finger position for each frame
    fpos = data[:,(1,2)].astype(np.float32)
    min_pos = np.min(fpos, 0)
    range_pos = np.ptp(fpos, 0)

    # scale the positions from -0.5 to 0.5
    scaled_pos = scale_pos(fpos, min_pos, range_pos)

    pos_struct = lambda: None
    pos_struct.frames = frames
    pos_struct.fpos = fpos
    pos_struct.min_pos = min_pos
    pos_struct.range_pos = range_pos
    pos_struct.scaled_pos = scaled_pos

    return pos_struct

def makeFeature(pos_struct, vs, index, dn, be, counter):
    ps = pos_struct

    if counter % report_interval == 0:
        print "Generating feature %d of %d" % (counter, ps.frames.size)

    vs.setFrame(int(ps.frames[index]))
    vs.grab()
    for j in range(dn.getLayerCount()): # run enough times to flush out the data from prev frames
        dn.doDestin(vs.getOutput())

    return (be.getBeliefsNumpy(be.getOutputSize()), ps.scaled_pos[index])

def createFeatures(dn, vs):
    print "Creating features..."
    be = pd.BeliefExporter(dn, 0)
    vs.setFrame(0)
    pos_struct = loadTruthLabels()

    # for training
    nn_train_features = []
    nn_train_lables = []

    # cross validation
    cv_features = []
    cv_labels = []

    m = pos_struct.frames.size

    # randomize the data
    ri = range(m) # = random inicies
    random.shuffle(ri)

    cv_size = int(m * cross_valid_frac)
    # create train features
    for i in xrange(m - cv_size):
        feature, label = makeFeature(pos_struct, vs, ri[i], dn, be, i)
        nn_train_features.append(feature)
        nn_train_lables.append(label)

    # create cross validation features
    for i in xrange(m - cv_size + 1, m):
        feature, label = makeFeature(pos_struct, vs, ri[i], dn, be, i)
        cv_features.append(feature)
        cv_labels.append(label)

    print "Finished creating features."

    return (np.matrix(nn_train_features),
             np.matrix(nn_train_lables),
             np.matrix(cv_features),
             np.matrix(cv_labels))

def trainDestin(dn, layers):
    dn.setIsTraining(False)

    for stage in xrange(layers):
        if stage > 0:
            dn.setLayerIsTraining(stage - 1, False)
        dn.setLayerIsTraining(stage, True)

        for i in xrange(destin_train_iterations):
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
    nn.save(nn_save_file)
    return nn

def meanSquaredError(actual, expected):
    return np.abs(actual - expected).mean()
    #return np.power((actual - expected),2).mean()

def checkAccuracy(neural_net, train_features, train_labels, cv_features, cv_labels):
    dummy, train_preds = neural_net.predict(train_features)
    dummy, cv_preds = neural_net.predict(cv_features)
    print "MSE Train:", meanSquaredError(train_preds, train_labels)
    print "MSE CV:", meanSquaredError(cv_preds, cv_labels)


def visualizePrediction(vs, nn, dn):
    dn.setIsTraining(False)

    pos_struct = loadTruthLabels()
    be = pd.BeliefExporter(dn,0)

    ratio_x = float(visual_width_x) / labels_max_x
    ratio_y = float(visual_height_y) / labels_max_y
    for i in xrange(pos_struct.frames.size):
        feature, coord = makeFeature(pos_struct, vs, i, dn, be, i)
        dummy, pred = nn.predict(feature.reshape(1,feature.size))

        pred_pos = unscale_pos(pred[0], pos_struct.min_pos, pos_struct.range_pos)

        x = int(pred_pos[0] * ratio_x)
        y = int(pred_pos[1] * ratio_y)
        expected_x = int(pos_struct.fpos[i][0] * ratio_x)
        expected_y = int(pos_struct.fpos[i][1] * ratio_y)

        vs.grab()
        image = vs.getOutputColorMatNumpy().reshape(destin_width,destin_width,3)
        resized = cv2.resize(image, (int(visual_width_x), int(visual_height_y)))
        cv2.circle(resized, (expected_x, expected_y), radius=5, color=(255,0,0), thickness=-1)
        cv2.circle(resized, (x, y), radius=5, color=(0,255,0), thickness=-1)
        cv2.imshow('frame',resized)
        wk = cv2.waitKey() # waits till key press
        if wk & 0xFF == ord('q'): # break if q is pressed
            break


## Script body ##
vs = pd.VideoSource(False, input_video)
vs.setSize(destin_width, destin_width)

if train_destin:
    dn = pd.DestinNetworkAlt(destin_width, layers, centroids, True)
    dn, vs = trainDestin(dn, layers)
else:
    dn = pd.DestinNetworkAlt(load_destin)
    dn.setIsTraining(False)

features, labels, cv_features, cv_labels = createFeatures(dn, vs)

if train_nn:
    nn = trainNN(features, labels)
else:
    nn = cv2.ANN_MLP()
    nn.load(nn_save_file)

checkAccuracy(nn, features, labels, cv_features, cv_labels)
visualizePrediction(vs, nn, dn)
