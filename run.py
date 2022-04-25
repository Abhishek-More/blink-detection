from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import sys
import os
import imutils
import random
import time
import dlib
import cv2

#CONSTANTS
options = ["Raise eyebrows", "Open Mouth", "Move Head Left", "Move Head Right"]
timeThres = 2


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

def isEyebrowRaised(gray_frame, rect):
    shape = predictor(gray_frame, rect)
    shape = face_utils.shape_to_np(shape)
    leftEye = shape[left_s:left_e]
    rightEye = shape[right_s:right_e]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    leftEyebrow = shape[left_e_start:left_e_end]
    lefteyebrowEBAR = eyebrow_aspect_ratio(leftEyebrow, leftEye) 
    leftebar = lefteyebrowEBAR
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    lefteyebrowHull = cv2.convexHull(leftEyebrow)
    cv2.drawContours(frame, [leftEyeHull], -1, (255, 0, 0), 1)
    cv2.drawContours(frame, [rightEyeHull], -1, (255, 0, 0), 1)
    cv2.drawContours(frame, [lefteyebrowHull], -1, (255, 0, 0), 1)

    print(leftebar)
    if leftebar > EYEBROW_AR_THRESH:
        print("Raised")
        cv2.putText(frame, "Eyebrow Raised!", (30,90),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
        return True
    else:
        return False

def isHeadLeft(gray_frame, rect):
    shape = predictor(gray_frame, rect)
    shape = face_utils.shape_to_np(shape)
    leftEye = shape[left_s:left_e]
    rightEye = shape[right_s:right_e]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(frame, [leftEyeHull], -1, (255, 0, 0), 1)
    cv2.drawContours(frame, [rightEyeHull], -1, (255, 0, 0), 1)

    if rightEye[0][0] < 200:
        return True
    else:
        return False

def isHeadRight(gray_frame, rect):
    shape = predictor(gray_frame, rect)
    shape = face_utils.shape_to_np(shape)
    leftEye = shape[left_s:left_e]
    rightEye = shape[right_s:right_e]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(frame, [leftEyeHull], -1, (255, 0, 0), 1)
    cv2.drawContours(frame, [rightEyeHull], -1, (255, 0, 0), 1)

    if leftEye[0][0] > 450:
        return True
    else:
        return False

def isMouthOpen(gray_frame, rect):
    shape = predictor(gray_frame, rect)
    shape = face_utils.shape_to_np(shape)
    mouth = shape[mStart:mEnd]
    mouthMAR = mouth_aspect_ratio(mouth)
    mar = mouthMAR
    mouthHull = cv2.convexHull(mouth)
    cv2.drawContours(frame, [mouthHull], -1, (255, 0, 0), 1)

    if mar > MOUTH_AR_THRESH:
        print("Open")
        cv2.putText(frame, "Mouth is Open!", (30,60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
        return True
    else:
        return False

BLINK_THRESHOLD = 0.19  # the threshold of the ear below which we assume that the eye is closed
MOUTH_AR_THRESH = 0.79
EYEBROW_AR_THRESH = 1.5
CONSEC_FRAMES_NUMBER = 2  # minimal number of consecutive frames with a low enough ear value for a blink to be detected

# get arguments from a command line
ap = argparse.ArgumentParser(description='Eye blink detection')
ap.add_argument("-p", "--shape-predictor", required=False, help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="", help="path to input video file")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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
score = 0
highScore = 0
commandStart = time.time()
funcs = [isEyebrowRaised, isMouthOpen, isHeadLeft, isHeadRight]

detected = False

num = random.randint(0, len(options)-1)
while True:

    started = False

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    frame = vs.read()

    cv2.putText(frame, "Perform: '" + options[num] + "' when you see the highlights", (30,20),
    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
    start = time.time()

    if(time.time() - commandStart > 2):
        detected = False
        started = True
        while ((not fileStream) or (frame is not None)) and (time.time() - start < timeThres):
            frame = imutils.resize(frame, width=640)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray_frame, 0)

            for rect in rects:
                if(funcs[num](gray_frame, rect)):
                    commandStart = time.time()
                    score += 1
                    if highScore < score:
                        highScore = score 
                    detected = True
                    break

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            frame = vs.read()

            if(detected):
                break

    if detected:
        cv2.putText(frame, "Good job!", (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
    if not detected and started:
        cv2.putText(frame, "Task Failed!", (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
        score = 0
        commandStart = time.time()

    if started:
        num = random.randint(0, len(options)-1)

    cv2.putText(frame, "Score: " + str(score), (30, 50),
    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

cv2.destroyAllWindows()
vs.stop()
