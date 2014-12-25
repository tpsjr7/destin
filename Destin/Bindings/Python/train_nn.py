#!/usr/bin/python
import cv2
import numpy as np
import pydestin as pd
import random
import datetime
import sklearn.svm
import scipy.sparse
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV

### Config ###

train_destin = False    # If true, train destin otherwise load from destin_save_file
create_features = False # if true, create features from destin otherwise load from features_save_file
train_predictor = False  # if true then train the predictor, otherwise load from predictor_save_file
calc_final_error = False # Run predictor on all train and test features and report the error
show_visualization = False # Show posistion prediction visualization on training video
show_visualization_live = False # Show posistion prediction with live webcam
create_learning_curve = False #

predictor_type="SVM" # ANN or SVM

destin_save_file = "train_predictor.dst"
features_save_file = "finger_features.npy"
predictor_save_file = "saves/predictor_%s_save.save" % (predictor_type)

# If true, tries training the NN multiple times to find best # of units
# when train_predictor is true
search_hidden_units = False

nn_hidden_units_search = [2,3,10,20,40,80,160]
nn_hidden_units = 20 # use this if search_hidden_units is False

#1  2 4  8  16 32  64
#4  8 16 32 64 128 256
centroids = [5,8,16,16,16]
#centroids = [5,5,5,5,5,5]

layers = len(centroids)

#when training samples were created, this was the video dimensions
labels_max_x = 512.0
labels_max_y = 512.0

# Size of the visualization window
visual_width_x = 512.0
visual_height_y = 512.0

# Size of the video fed to DeSTIN
destin_video_width = 4 * 2**(layers - 1)

# Fraction of feature vectors reserved for cross validation
cross_valid_frac = 0.3

# Training iterations per destin layer
destin_train_iterations = 900
jump_speed = 7

position_train_data="./really_big_lables.txt"
#position_train_data="./large_fingers.txt"
#position_train_data="./finger_data.txt"
#input_video = "./finger.mov"
#input_video = "./large_finger.mov"
input_video = "./really_big_finger.mov"

report_interval = 20

### Global Vars ###
grid_search_clf = None

### Functions ###

class ANN:
    def __init__(self, features, labels, config_dic):

        hidden_units = config_dic.get("hidden_units", nn_hidden_units)
        self.nn = self.createNN(features, labels, hidden_units)
        print self.nn

    def createNN(self, features, labels, hidden_units):
        input_units = np.size(features,1) # length of destin belief feature vector
        output_units = np.size(labels, 1) # should be just 2 for x,y
        nn_layers = np.array([input_units, hidden_units, output_units])
        nn = cv2.ANN_MLP(nn_layers, cv2.ANN_MLP_SIGMOID_SYM, 1.0, 1.0)
        print "Created NN with hidden units:", hidden_units
        return nn

    def makeParams(self):
        term_crit = pd.cvTermCriteria(pd.CV_TERMCRIT_ITER | pd.CV_TERMCRIT_EPS, 20000, 0.0000001)
        bp_dw_scale = 0.1
        bp_moment_scale = 0.1
        p = pd.CvANN_MLP_TrainParams(term_crit, int(cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP), bp_dw_scale, bp_moment_scale)
        print dir(p)
        pd.dumpParams(p)
        return p

    def train(self, features, labels):
        self.nn.train(features, labels, None, None, self.makeParams())

    def predict(self, features):
        dummy, pred = self.nn.predict(features)
        return pred

    def save(self, filename):
        self.nn.save(filename)

    def load(self, filename):
        self.nn = cv2.ANN_MLP()
        self.nn.load(filename)

class SVM:
    def __init__(self):
        #params = {'C': 3, 'epsilon': 0.01, 'gamma': 0.03, 'kernel': 'rbf'}
        self.svm_x = sklearn.svm.SVR(C=30,epsilon=0.01, gamma=0.03, kernel='rbf')
        self.svm_y = sklearn.svm.SVR(C=30,epsilon=0.01, gamma=0.03, kernel='rbf')
        #{'C': 30, 'epsilon': 0.01, 'gamma': 0.03, 'kernel': 'rbf'}
        #self.svm_x.set_params(params)
        #self.svm_y.set_params(params)

    def train(self, features, labels):
        assert labels.shape[1] == 2 # make sure has just 2 columns, x and y positoins
        x = np.asarray(labels[:,0]).reshape(-1,) # massage it into the expected format
        y = np.asarray(labels[:,1]).reshape(-1,)
        print self.svm_x.fit(features, x)
        print self.svm_y.fit(features, y)

    def predict(self, X):
        x = self.svm_x.predict(X).reshape(-1,1) # make into column vector Mx1
        y = self.svm_y.predict(X).reshape(-1,1)
        return np.concatenate((x,y),1) #make into 2 column matrix, Mx2

    def save(self, filename):
        svms = (self.svm_x, self.svm_y)
        joblib.dump(svms, predictor_save_file)

    def load(self, filename):
        self.svm_x, self.svm_y = joblib.load(predictor_save_file)

def createPredictor(name, features, labels, config_dict = {}):
    if name=="SVM":
        return SVM()
    elif name=="ANN":
        return ANN(features, labels, config_dict)
    else:
        raise RuntimeError("Unknown predictor type " + str(name))

def normalize(array):
    mean = array.mean(0)
    norm = array - mean
    std = norm.std(0)
    norm = norm / std
    return (norm, mean, std)

def un_normalize(array, mean, std):
    out = array * std + mean
    return out

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
    print "Loading truth labels"
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

def createFeatures(dn, vs, pos_struct):
    print "Creating features..."
    be = pd.BeliefExporter(dn, 0)
    print "Feature dimension is", be.getOutputSize()
    vs.setFrame(0)

    # for training
    predictor_train_features = []
    predictor_train_lables = []

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
        predictor_train_features.append(feature)
        predictor_train_lables.append(label)

    # create cross validation features
    for i in xrange(m - cv_size + 1, m):
        feature, label = makeFeature(pos_struct, vs, ri[i], dn, be, i)
        cv_features.append(feature)
        cv_labels.append(label)

    print "Finished creating features."

    ret = (np.array(predictor_train_features),
             np.array(predictor_train_lables),
             np.array(cv_features),
             np.array(cv_labels))

    np.save(features_save_file, ret)

    return ret

def trainDestin(dn, layers):
    dn.setIsTraining(False)

    for stage in xrange(layers):
        if stage > 0:
            dn.setLayerIsTraining(stage - 1, False)
        dn.setLayerIsTraining(stage, True)

        for i in xrange(destin_train_iterations):
            vs.setFrame(vs.getFrame() + jump_speed)
            if vs.grab():
                dn.doDestin(vs.getOutput())
                if i % 10 == 0:
                    print "S: %d I: %d Q: %f" % (stage, i, dn.getQuality(stage))

    dn.setIsTraining(False)
    dn.save(destin_save_file)

    return (dn, vs)

def trainPredictor(features, labels):
    print "Starting predictor training..."

    predictor = createPredictor(predictor_type, features, labels)

    predictor.train(features, labels)
    print "Finished predictor training"
    predictor.save(predictor_save_file)
    return predictor

def meanAbsError(actual, expected):
    return np.abs(actual - expected).mean()

def meanSquaredError(actual, expected):
    return np.power((actual - expected),2).mean()

def checkAccuracy(predictor, train_features, train_labels, cv_features, cv_labels):
    print "Checking accuracy..."
    print "Checking train accuracy..."
    train_preds = predictor.predict(train_features)
    print "Checking CV accuracy..."
    cv_preds = predictor.predict(cv_features)
    train_mse = meanSquaredError(train_preds, train_labels)
    cv_mse =  meanSquaredError(cv_preds, cv_labels)
    print "MSE Train:", train_mse
    print "MSE CV:", cv_mse

    train_mae = meanAbsError(train_preds, train_labels)
    cv_mae = meanAbsError(cv_preds, cv_labels)

    print "MAE Train:", train_mae
    print "MAE CV:", cv_mae

    return (train_mse, cv_mse)

def visualizePrediction(vs, predictor, dn, pos_struct):
    print "Now displaying visualization window."
    dn.setIsTraining(False)

    be = pd.BeliefExporter(dn,0)

    ratio_x = float(visual_width_x) / labels_max_x
    ratio_y = float(visual_height_y) / labels_max_y
    for i in xrange(pos_struct.frames.size):
        feature, coord = makeFeature(pos_struct, vs, i, dn, be, i)
        pred = predictor.predict(feature.reshape(1,feature.size))

        pred_pos = unscale_pos(pred[0], pos_struct.min_pos, pos_struct.range_pos)

        x = int(pred_pos[0] * ratio_x)
        y = int(pred_pos[1] * ratio_y)
        expected_x = int(pos_struct.fpos[i][0] * ratio_x)
        expected_y = int(pos_struct.fpos[i][1] * ratio_y)

        vs.grab()
        image = vs.getOutputColorMatNumpy().reshape(destin_video_width,destin_video_width,3)
        resized = cv2.resize(image, (int(visual_width_x), int(visual_height_y)))
        cv2.circle(resized, (expected_x, expected_y), radius=5, color=(255,0,0), thickness=-1)
        cv2.circle(resized, (x, y), radius=5, color=(0,255,0), thickness=-1)
        cv2.imshow('frame',resized)
        wk = cv2.waitKey() # waits till key press
        if wk & 0xFF == ord('q'): # break if q is pressed
            cv2.destroyAllWindows()
            break

def visualizeLive(predictor, dn, pos_struct):
    print "Running live visualization..."
    wc = pd.VideoSource(True, "")
    dn.setIsTraining(False)
    wc.setSize(destin_video_width, destin_video_width)
    be = pd.BeliefExporter(dn,0)
    ratio_x = float(visual_width_x) / labels_max_x
    ratio_y = float(visual_height_y) / labels_max_y

    while True:
        if wc.grab():
            dn.doDestin(wc.getOutput())

            feature = be.getBeliefsNumpy(be.getOutputSize())
            pred = predictor.predict(feature.reshape(1,feature.size))

            pred_pos = unscale_pos(pred[0], pos_struct.min_pos, pos_struct.range_pos)

            x = int(pred_pos[0] * ratio_x)
            y = int(pred_pos[1] * ratio_y)

            image = wc.getOutputColorMatNumpy().reshape(destin_video_width,destin_video_width,3)
            resized = cv2.resize(image, (int(visual_width_x), int(visual_height_y)))
            cv2.circle(resized, (x, y), radius=5, color=(255,0,0), thickness=-1)
            cv2.imshow('frame',resized)
            wk = cv2.waitKey(5)
            if wk & 0xFF == ord('q'): # break if q is pressed
                cv2.destroyAllWindows()
                break

def search(possible_hidden_units, X, y, X_cv, y_cv):
    """
        Searches for the best hidden units value by testing multiple neural networks
    """
    stats = []
    min_cv_err = float("inf")
    best_nn = None
    best_index = 0
    for index, hu in enumerate(possible_hidden_units):
        temp_nn = createPredictor(predictor_type, X, y, {'hidden_units': hu})
        train_err, cv_err = checkAccuracy(temp_nn, X, y, X_cv, y_cv)
        stats.append([train_err, cv_err])
        if cv_err < min_cv_err:
            best_nn = temp_nn
            min_cv_err = cv_err
            best_index = index

    return (best_nn, best_index, stats)

def createLearningCurve(features, labels, cv_features, cv_labels):

    print "Creating learning curve..."
    steps = 10
    start = 10
    end = features.shape[0]
    step_size =(end - start) / steps

    errors = []
    count = 0
    for size in xrange(start, end, step_size):
        count = count + 1
        print "Step %d of %d. Using %d of %d training samples." %(count, steps, size, end)
        feats = features[0:size,:]
        labs = labels[0:size,:]
        predictor = trainPredictor(feats, labs)
        train_err, cv_err = checkAccuracy(predictor, feats, labs, cv_features, cv_labels)
        errors.append((train_err, cv_err))

    return errors

def gridSearch(features, labels):
    global grid_search_clf
    tuned_parameters = [
        {'kernel': ['rbf'],
        'gamma': [.03, .01, .003],
        'epsilon':[.01,.03,.1],
        'C': [15, 30, 60]}]

    score = 'mean_squared_error'
    clf = GridSearchCV(sklearn.svm.SVR(C=1), tuned_parameters, cv=2, scoring=score, n_jobs=4)

    scores = ['mean_squared_error']
    x_labels = np.asarray(labels[:,0]).reshape(-1,)
    #y_labels = np.asarray(labels[:,1]).reshape(-1,)
    clf.fit(features, x_labels)
    grid_search_clf = clf
    return clf


## Script body ##
print  "Creating video source."
vs = pd.VideoSource(False, input_video)
vs.setSize(destin_video_width, destin_video_width)
pos_struct = loadTruthLabels()

if train_destin:
    dn = pd.DestinNetworkAlt(destin_video_width, layers, centroids, True)
    dn, vs = trainDestin(dn, layers)
else:
    print "Loading destin network"
    dn = pd.DestinNetworkAlt(destin_save_file)
    dn.setIsTraining(False)

if create_features:
    features, labels, cv_features, cv_labels = createFeatures(dn, vs, pos_struct)
else:
    print "Loading features..."
    features, labels, cv_features, cv_labels = np.load(features_save_file)
    print "Finished loading features..."

if train_predictor:
    if search_hidden_units and predictor_type=="ANN" :
        best_nn, best_index, stats = search(nn_hidden_units_search, features,
            labels, cv_features, cv_labels)

        for i,s in enumerate(stats):
            print "HU:",nn_hidden_units_search[i], "Train err:", s[0], "CV err:", s[1]

        print "Best CV err:", stats[best_index][1], "Hidden units:", nn_hidden_units_search[i]
        predictor = best_nn
    else:
        predictor = trainPredictor(features, labels)

else:
    predictor = createPredictor(predictor_type, features, labels)
    print "Loading predictor from", predictor_save_file
    predictor.load(predictor_save_file)

if calc_final_error:
    checkAccuracy(predictor, features, labels, cv_features, cv_labels)

if show_visualization:
    visualizePrediction(vs, predictor, dn, pos_struct)

if show_visualization_live:
    visualizeLive(predictor, dn, pos_struct)

if create_learning_curve:
    learning_curve = createLearningCurve(features, labels, cv_features, cv_labels)
