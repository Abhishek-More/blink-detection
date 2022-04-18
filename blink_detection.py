from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2


# compute the Eye Aspect Ratio (ear),
# which is a relation of the average vertical distance between eye landmarks to the horizontal distance
def eye_aspect_ratio(eye):
    vertical_dist = dist.euclidean(eye[1], eye[5]) + dist.euclidean(eye[2], eye[4])
    horizontal_dist = dist.euclidean(eye[0], eye[3])
    ear = vertical_dist / (2.0 * horizontal_dist)
    return ear

def mouth_aspect_ratio(mouth):
	# compute the euclidean distances between the two sets of
	# vertical mouth landmarks (x, y)-coordinates
	A = dist.euclidean(mouth[2], mouth[10]) # 51, 59
	B = dist.euclidean(mouth[4], mouth[8]) # 53, 57

	# compute the euclidean distance between the horizontal
	# mouth landmark (x, y)-coordinates
	C = dist.euclidean(mouth[0], mouth[6]) # 49, 55

	# compute the mouth aspect ratio
	mar = (A + B) / (2.0 * C)

	# return the mouth aspect ratio
	return mar

def eyebrow_aspect_ratio(eyebrow, eye):
    A = dist.euclidean(eyebrow[2], eye[1]);
    C = dist.euclidean(eyebrow[0], eyebrow[3]);

    ebar = (A) / C
    return ebar


BLINK_THRESHOLD = 0.19  # the threshold of the ear below which we assume that the eye is closed
MOUTH_AR_THRESH = 0.79
EYEBROW_AR_THRESH = 1.5
CONSEC_FRAMES_NUMBER = 2  # minimal number of consecutive frames with a low enough ear value for a blink to be detected

# get arguments from a command line
ap = argparse.ArgumentParser(description='Eye blink detection')
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="", help="path to input video file")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# choose indexes for the left and right eye
(left_s, left_e) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_s, right_e) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = (49, 68)
(left_e_start, left_e_end) = (18,22)
(right_e_start, right_e_end) = (23,27)

# start the video stream or video reading from the file
video_path = args["video"]
if video_path == "":
    vs = VideoStream(src=0).start()
    print("[INFO] starting video stream from built-in webcam...")
    fileStream = False
else:
    vs = FileVideoStream(video_path).start()
    print("[INFO] starting video stream from a file...")
    fileStream = True
time.sleep(1.0)

counter = 0
total = 0
alert = False
start_time = 0
frame = vs.read()
leftebar = 0

# loop over the frames of video stream:
# grab the frame, resize it, convert it to grayscale
# and detect faces in the grayscale frame
while (not fileStream) or (frame is not None):
    frame = imutils.resize(frame, width=640)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray_frame, 0)
    ear = 0
    ebar = 0
    # loop over the face detections:
    # determine the facial landmarks,
    # convert the facial landmark (x, y)-coordinates to a numpy array,
    # then extract the left and right eye coordinates,
    # and use them to compute the average eye aspect ratio for both eyes
    for rect in rects:
        shape = predictor(gray_frame, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[left_s:left_e]
        rightEye = shape[right_s:right_e]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        mouth = shape[mStart:mEnd]
        leftEyebrow = shape[left_e_start:left_e_end]
        mouthMAR = mouth_aspect_ratio(mouth)
        lefteyebrowEBAR = eyebrow_aspect_ratio(leftEyebrow, leftEye) 
        mar = mouthMAR
        leftebar = lefteyebrowEBAR
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
		# compute the convex hull for the mouth, then
		# visualize the mouth
        mouthHull = cv2.convexHull(mouth)
        lefteyebrowHull = cv2.convexHull(leftEyebrow)
        cv2.drawContours(frame, [leftEyeHull], -1, (255, 0, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (255, 0, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (255, 0, 0), 1)
        cv2.drawContours(frame, [lefteyebrowHull], -1, (255, 0, 0), 1)

        # if the eye aspect ratio is below the threshold, increment counter
        # if the eyes are closed longer than for 2 secs, raise an alert
        if ear < BLINK_THRESHOLD:
            counter += 1
            if start_time == 0:
                #print("BLINKED AT " + str(time.time()))
                start_time = time.time()
            else:
                end_time = time.time()
                if end_time - start_time > 2: alert = True
        else:
            if counter >= CONSEC_FRAMES_NUMBER:
                total += 1
            counter = 0
            start_time = 0
            alert = False
        if mar > MOUTH_AR_THRESH:
            cv2.putText(frame, "Mouth is Open!", (30,60),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
        if leftebar > EYEBROW_AR_THRESH:
            cv2.putText(frame, "Eyebrow Raised!", (30,90),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)



    # draw the total number of blinks and EAR value
    cv2.putText(frame, "Blinks: {}".format(total), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    #cv2.putText(frame, "EAR: {:.2f}".format(ear), (500, 30),
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Blinks: {}".format(leftebar), (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    #cv2.putText(frame, "MAR: {:.2f}".format(mar), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if alert:
        cv2.putText(frame, "ALERT!", (150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    frame = vs.read()

cv2.destroyAllWindows()
vs.stop()
