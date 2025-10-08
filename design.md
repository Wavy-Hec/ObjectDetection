Real‑Time Object Tracking Project
Objective

This project aims to build a real‑time object‑tracking system that detects and tracks objects (people and cars) in video streams. A pre‑trained object detector (YOLOv5s) identifies objects in each frame, while a tracking algorithm (SORT – Simple Online and Realtime Tracking) assigns consistent IDs across frames. The system should run fast enough for live demonstrations and support recording for evaluation and presentation.

Scope and Assumptions

Target objects: people and cars, as detected by the YOLOv5s model. Other classes will be ignored.

Detector: YOLOv5s from the Ultralytics library, using a 640×480 input resolution and a confidence threshold of 0.25. Models are pre‑trained on the COCO dataset and loaded via ultralytics.YOLO.

Tracker: SORT (Kalman filter–based), which uses bounding boxes and IOU matching to maintain object identities. The tracker assigns a unique ID to each detected object and keeps it consistent across frames.

Performance goals: Achieve >30 FPS on a typical laptop GPU (e.g., RTX 3050 or better). Performance on CPU will be slower; a GPU is recommended.

Hardware constraints: A conda environment with Python 3.10 will be used. Dependencies include torch, ultralytics, opencv‑python, sort, filterpy, and motmetrics. Assumes a CUDA‑capable GPU for acceleration.

Evaluation data: A small set of public videos (e.g., from the MOT16 dataset or recorded street scenes) will be used for testing. Ground‑truth annotations will enable the calculation of tracking metrics such as MOTA and ID switches.

Output format: Each frame will display bounding boxes with class labels and unique IDs. The system will save processed videos and log metrics for evaluation. A CLI will allow users to specify input videos and output paths.

Design Details
Input and Output

Input: Video stream (file or webcam) with 3‑channel RGB frames. Frames will be resized to 640×480 before detection to standardise processing time.

Detection output: A list of detections, where each detection consists of a bounding box [x1, y1, x2, y2], a class label ("person" or "car"), and a confidence score between 0 and 1.

Tracking output: A list of tracks, where each track includes the current bounding box, the assigned ID, and associated class and confidence. Tracks may disappear if the object leaves the frame or is occluded for too long.

Visualisation: Bounding boxes and IDs will be drawn onto frames using OpenCV. A unique colour will be associated with each track ID for clarity.

System Components

Detector module (detector.py):

Loads the YOLOv5s model from the Ultralytics library.

Provides a detect(frame) function that runs inference on an input frame and returns detections (boxes, labels, scores) filtered by confidence threshold.

Uses CPU or GPU depending on availability.

Tracker module (tracker.py):

Implements the SORT algorithm using a Kalman filter to estimate object trajectories.

Includes functions to initialise tracks, update them with new detections, and delete tracks that have not been updated for a given number of frames.

Returns updated tracks with unique IDs.

Integration (main.py):

Loads a video stream (file or webcam) via OpenCV.

Runs detection on each frame using detector.py and updates tracks using tracker.py.

Draws bounding boxes and IDs onto the frame and displays them in real time.

Records the processed video to disk and logs performance metrics (FPS, detection count, track count).

Evaluation (evaluation.py, optional):

Computes metrics such as MOTA (Multiple Object Tracking Accuracy), MOTP (Precision), and the number of ID switches using the motmetrics library.

Generates summary statistics and plots for reporting.

Success Metrics

The project will be considered successful if it meets the following criteria:

Real‑time performance: The system processes at least 30 frames per second on a GPU; on CPU, a lower FPS is acceptable but the system should not drop below 15 FPS.

Tracking accuracy: On a small evaluation set, the system achieves a MOTA > 0.80 and fewer than 5 ID switches per minute of video.

Modularity: Code is organised into separate modules (detector.py, tracker.py, main.py) with clear interfaces. Unit tests cover key functionality.

Documentation: design.md captures the design decisions and assumptions. A README.md explains how to set up the environment, run the system, and interpret results.

Risks and Mitigations

Performance on CPU: If a GPU is unavailable, inference speed will be slower. Mitigation: Use the smaller YOLOv5n model or reduce frame resolution to 416×416.

ID switches: SORT can produce ID switches during occlusions or when objects cross paths. Mitigation: Experiment with Deep SORT (appearance embeddings) if time permits.

Dataset bias: The COCO‑trained model may perform poorly on unusual angles or lighting conditions. Mitigation: Validate on a diverse set of videos; adjust confidence threshold accordingly.