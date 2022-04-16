# Installation guide
## I. Local installation
1. Clone repository:
```
git clone https://github.com/nWhovian/blink-detection.git
```
2. Move to the cloned folder and install requirements:
```
cd path/to/blink-detection
pip install -r requirements.txt
```
3. Download the dlib’s pre-trained facial landmark detector
```
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=17YdgpbtvxWvPhRHIkuSwjlp9nFSBKv3M' -O shape_predictor_68_face_landmarks.dat
```
4. (optional) You can download a video with my face and use it for testing
```
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=12gbn8y0qt1aIFOw9R8lmVX2LBPCuinJB' -O video.mp4
```
5. Run the script with video file argument
```
python ./blink_detection.py --shape-predictor shape_predictor_68_face_landmarks.dat --video -path/to/video
```
or with web camera streaming
```
python ./blink_detection.py --shape-predictor shape_predictor_68_face_landmarks.dat
```
You can see the video examples of blink detection using here:
```
https://drive.google.com/file/d/1XwlD3P_TnJui66ysGpBa7b7Zet6SUYea/
```
