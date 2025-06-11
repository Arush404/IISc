Exercise Pose Detection Suite
This repository provides real-time computer vision applications for exercise and yoga pose detection using a webcam. The suite includes three main scripts:

curl.py: Counts bicep curl repetitions and checks form.

pushup.py: Counts push-up repetitions and provides feedback on body alignment.

trik4.py: Guides the user through the Trikonasana (Triangle) yoga pose, checking pose correctness and providing feedback.

Each script uses MediaPipe for pose estimation and OpenCV for video processing and visualization.

Contents
Requirements

How It Works

1. Bicep Curl Counter (curl.py)

2. Push-Up Counter (pushup.py)

3. Trikonasana Yoga Pose Sequencer (trik4.py)

Algorithms and Logic

Usage

Customization

Requirements
Python 3.7+

OpenCV

MediaPipe

NumPy

Install dependencies with:

bash
pip install opencv-python mediapipe numpy
How It Works
1. Bicep Curl Counter (curl.py)1
Uses MediaPipe Pose to detect key landmarks (shoulder, elbow, wrist, hip) for both arms.

Calculates the elbow angle to determine arm extension/flexion.

Counts a repetition when the arm transitions from extended (down) to flexed (up) and back.

Checks if the upper arm remains vertical (good form) using the angle between hip, shoulder, and elbow.

Displays real-time feedback, rep count, and form warnings on the video stream.

2. Push-Up Counter (pushup.py)2
Detects body landmarks (shoulders, elbows, wrists, hips, ankles) using MediaPipe Pose.

Calculates angles at the elbows and hips to determine push-up stages:

Down: Elbow angle between 70-100°, hip angle between 160-200°.

Up: Elbow angle above 160°, hip angle in the same range.

Increments the counter when a full push-up is completed (down → up).

Provides visual feedback on body alignment ("Perfect Long Line Body") and displays a progress bar.

Draws pose skeleton and angles for user reference.

3. Trikonasana Yoga Pose Sequencer (trik4.py)3
Implements a class-based system to analyze and guide yoga poses, starting with Trikonasana.

For Trikonasana:

Checks five key elements: front leg straightness, back leg straightness, torso side bend, top arm extension, and arm alignment.

Each element is evaluated using geometric angle calculations between relevant landmarks.

Provides specific feedback for each element (✓ for correct, ⚠ for adjustments).

Tracks pose stability over time and requires the user to hold the correct pose for a set duration (default: 5 seconds).

Displays instructions, feedback, and a timer on the video stream.

The sequencer can be extended to include additional yoga poses.

Algorithms and Logic
Angle Calculation
All scripts use a geometric method to calculate the angle at a joint given three points 
a
a, 
b
b, 
c
c (joint at 
b
b):

θ
=
∣
arctan
⁡
2
(
c
y
−
b
y
,
c
x
−
b
x
)
−
arctan
⁡
2
(
a
y
−
b
y
,
a
x
−
b
x
)
∣
θ=∣arctan2(c 
y
 −b 
y
 ,c 
x
 −b 
x
 )−arctan2(a 
y
 −b 
y
 ,a 
x
 −b 
x
 )∣
or

θ
=
arccos
⁡
(
(
a
−
b
)
⋅
(
c
−
b
)
∥
a
−
b
∥
⋅
∥
c
−
b
∥
)
θ=arccos( 
∥a−b∥⋅∥c−b∥
(a−b)⋅(c−b)
 )
Angles are used to determine joint positions and to check for correct form or pose.

Repetition Counting

For curls and push-ups, the code tracks the stage ("up"/"down") based on angle thresholds.

A repetition is counted when the movement transitions through the full range (e.g., arm fully extended to fully flexed and back).

Form and Feedback

For both exercise and yoga, the code checks for proper alignment (e.g., vertical upper arm, straight legs).

Visual and textual feedback is overlaid on the video to guide the user in real-time.

Pose Stability and Sequencing (Yoga)

For yoga, the code tracks if the pose is held correctly for a required duration, using a frame-based stability check to avoid false positives from momentary misalignment.

Usage
Clone the repository and navigate to the folder.

Run the desired script:

bash
python curl.py      # For bicep curl counting
python pushup.py    # For push-up counting
python trik4.py     # For Trikonasana yoga pose guidance
Follow on-screen instructions and feedback.

Press q to quit at any time.

Customization
Add More Exercises or Poses:
Extend the logic in each script to support more movements by defining new angle checks and feedback.

Tune Thresholds:
Adjust angle thresholds in the scripts for sensitivity to different body types or standards.

Enhance Feedback:
Add audio feedback, logging, or more detailed analytics as needed.

References
MediaPipe Pose Documentation

OpenCV Documentation

