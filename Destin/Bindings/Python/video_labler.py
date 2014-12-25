import numpy as np
import cv2
import sys, getopt

# This script generates training data from a video.
# Each frame, you click where the object is, it outputs the frame# and the coordinates where you clicked.
# Press n to skip 5 frames. Press q to quit.
# You can manually call the record_cam function to record from your webcam into a new output video.


cap = None
out_features = None
last_shown_frame = 0
mode = "l" #l=label, c=capture

is_paused = False
def showFrame(n=1):
    global last_shown_frame

    # Capture frame-by-frame
    for i in xrange(n - 1):
        ret, frame = cap.read()


    frame_num = int(round(cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)))
    print "Frame: %d" % (frame_num)

    last_shown_frame = frame_num
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',gray)
    cv2.waitKey(1)

def callback(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print "F: %d, X: %d, Y: %d" % (last_shown_frame, x,y)
        out_features.write("%d %d %d\n" % (last_shown_frame, x, y))
        out_features.flush()
        showFrame()

def capture():
    while(True):
        wk = cv2.waitKey(1)
        if wk & 0xFF == ord('n'):
            showFrame(5)
        elif wk & 0xFF == ord('q'):
            break

def record_cam():
    vc = cv2.VideoCapture(0)

    #width = int(vc.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    #height = int(vc.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    record_frames = 0
    width = 512
    height = 512

    fcc = cv2.cv.CV_FOURCC('m','p','4','v')
    fps = 15
    vw = cv2.VideoWriter("theoutput.mov", fcc, fps, (width,height))

    count = 0
    is_paused = True # start in paused mode
    while True:
        ret, frame = vc.read()
        if frame == None:
            continue

        resized = cv2.resize(frame, (width,height))
        cv2.imshow('frame',resized)

        if record_frames > 0:
            record_frames = record_frames - 1
            if record_frames == 0:
                is_paused = True
                print "Paused"

        if not is_paused:
            vw.write(resized)
            count = count + 1
            if count % 10 == 0:
                print "Count: %d" % (count)

        wk = cv2.waitKey(1)
        if wk & 0xFF == ord('q'): # break if q is pressed
            cv2.destroyAllWindows()
            break
        elif wk & 0xFF == ord('p'): #pause
            is_paused = not is_paused
            if is_paused:
                print "Paused"
            else:
                print "Un-paused"
        elif wk & 0xFF == ord('s'):  # record a set amount of frames then pause
            record_frames = 30
            is_paused = False

    vw.release()

# tests how fast it can play a video shuffled
def playShuffle():
    vc = cv2.VideoCapture("finger.mov")
    ri = range(100)
    import random
    random.shuffle(ri)
    for i in ri:
        vc.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, i)
        ret, frame = vc.read()
        if frame != None:
            cv2.imshow('frame2', frame)
            cv2.waitKey(1)

def labler():
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame',callback)
    showFrame()
    capture()

def usage():
    print "%s -v <video> -f <startframe> -o <outputfile>" % (sys.argv[0])
    return

def main():
    global cap, out_features, mode
    opts, args = getopt.getopt(sys.argv[1:],"hv:f:o:m:",["video=","frame=","out=","mode="])

    start_frame = 0
    for opt, arg in opts:
        if opt == "-h":
           usage()
           sys.exit()
        elif opt in ("-v", "--video"):
            cap = cv2.VideoCapture(arg)
            print "Frame Count: %d" % (cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        elif opt in ("-o", "--out"):
            out_features = open(arg, 'a')
        elif opt in ("-f", "--frame"):
            start_frame = float(arg)
        elif opt in ("-m", "--mode"):
            mode = arg

    if mode=="l" and None in [out_features, cap]:
        print "Missing a required option."
        sys.exit()

    if mode=="l":
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, start_frame)
        labler()
    elif mode=="c":
        record_cam()
    else:
        print "Unknown mode %s" % (mode)



if __name__ == "__main__":
    main()


#playShuffle()


#record_cam(3000)

# When everything done, release the capture
#cap.release()
#cv2.destroyAllWindows()
