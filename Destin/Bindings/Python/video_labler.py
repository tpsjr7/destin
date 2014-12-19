import numpy as np
import cv2


# This script generates training data from a video.
# Each frame, you click where the object is, it outputs the frame# and the coordinates where you clicked.
# Press n to skip 5 frames. Press q to quit.
# You can manually call the record_cam function to record from your webcam into a new output video.


cap = cv2.VideoCapture("./finger.mov")
out_features = open("output.txt", 'w')

print "Frame Count: %d" % (cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

last_frame_num = -1
def showFrame(n=1):
    global last_frame_num
    # Capture frame-by-frame
    for i in xrange(n - 1):
        ret, frame = cap.read()


    frame_num = int(round(cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)))
    print "Frame: %d" % (frame_num)

    out_features.write("%d " % (frame_num))

    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',gray)
    cv2.waitKey(1)

def callback(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print "X: %d, Y: %d" % (x,y)
        out_features.write("%d %d\n" % (x,y) )
        out_features.flush()
        showFrame()

cv2.namedWindow('frame')
cv2.setMouseCallback('frame',callback)

showFrame()

def capture():
    while(True):
        wk = cv2.waitKey(1)
        if wk & 0xFF == ord('n'):
            showFrame(5)
        elif wk & 0xFF == ord('q'):
            break

def record_cam(frames = 300):
	vc = cv2.VideoCapture(0)

	#width = int(vc.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
	#height = int(vc.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

	width = 512
	height = 512

	fcc = cv2.cv.CV_FOURCC('m','p','4','v')
	fps = 15
	vw = cv2.VideoWriter("theoutput.mov", fcc, fps, (width,height))

	for x in xrange(frames):
		ret, frame = vc.read()
		if(frame != None):
			resized = cv2.resize(frame, (width,height))
			#gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
			vw.write(resized)
			cv2.imshow('frame',resized)
    		cv2.waitKey(1)
	vw.release()

#record_cam()
capture()
# When everything done, release the capture
#cap.release()
#cv2.destroyAllWindows()
