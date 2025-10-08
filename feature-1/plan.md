Project Plan: Real‑Time Object Tracking

This plan breaks the project into small, manageable tasks. Each task includes a description, dependencies, and acceptance criteria to verify completion.

Task 1 – Environment Setup

Description: Set up a clean conda environment (object_tracking) with Python 3.10 and install all necessary libraries. This isolates dependencies and ensures reproducibility.

Dependencies: None (should be done before coding).

Steps:

conda create -n object_tracking python=3.10 and conda activate object_tracking.

Install packages: pip install torch torchvision ultralytics opencv-python filterpy motmetrics sort.

Verify installation: run Python and import each package without errors.

Acceptance criteria: All packages import correctly; pip list shows installed dependencies.

Task 2 – Repository Structure

Description: Create the project directory with documentation files and the src/ folder for code.

Dependencies: Task 1 (conda environment ready).

Steps:

Use git init to initialise a repository (if not already done).

Create files: design.md (design overview), plan.md (this plan), README.md (setup instructions), and a src/ directory.

Inside src/, create placeholder files for detector.py, tracker.py, main.py, and evaluation.py (optional).

Acceptance criteria: Directory structure exists with all specified files; git status shows them as untracked or added.

Task 3 – Detection Module (detector.py)

Description: Implement a detection module that loads the YOLOv5s model and performs inference on frames.

Dependencies: Task 2 (project structure) and Task 1 (dependencies installed).

Steps:

In detector.py, import the Ultralytics YOLO class and load the YOLOv5s model.

Implement a detect(frame: np.ndarray) -> List[Detection] function that:

Converts the frame to the required format (RGB, 640×480).

Runs inference and filters detections by confidence threshold (e.g., 0.25).

Returns bounding boxes, class labels, and confidence scores for person and car classes only.

Include a simple test section or unit test to verify that the model detects objects on a sample image.

Acceptance criteria: When called on a test image, detect() returns a list of detections with bounding boxes and class labels. Unit test passes.

Task 4 – Tracking Module (tracker.py)

Description: Implement the SORT tracking algorithm to assign unique IDs to detected objects across frames.

Dependencies: Task 2 (project structure) and Task 1 (dependencies installed).

Steps:

Create a Track class or structure to hold state (bounding box, ID, age, and other attributes).

Implement a Tracker class that:

Maintains a list of active tracks.

Updates tracks with new detections using IOU matching and a Kalman filter (or use an existing SORT library to handle details).

Creates new tracks for unmatched detections and removes stale tracks.

Provide a update(detections: List[Detection]) -> List[Track] method that returns updated tracks.

Acceptance criteria: A simple test with synthetic bounding boxes shows that the tracker assigns stable IDs across frames when detections move smoothly. Unit test passes.

Task 5 – Integration Pipeline (main.py)

Description: Combine detection and tracking modules into an end‑to‑end pipeline that processes video frames.

Dependencies: Tasks 3 (detection) and 4 (tracking).

Steps:

Load a video stream using OpenCV (either a file or the default webcam).

For each frame:

Call detector.detect() to get detections.

Pass detections to tracker.update() to get tracks.

Draw bounding boxes and IDs on the frame using OpenCV.

Display the frame in a window and optionally write it to an output file.

Compute and print the FPS every few frames.

Handle end‑of‑video and user interruption gracefully.

Acceptance criteria: The script runs on a sample video and produces a processed video with bounding boxes and IDs. Real‑time FPS is printed to the console.

Task 6 – Unit Testing and Validation

Description: Write tests and evaluation routines to ensure modules behave as expected and to quantify tracking performance.

Dependencies: Tasks 3, 4, and 5.

Steps:

Use Python’s unittest or pytest to test individual functions: detection returns expected output types; tracker maintains consistent IDs.

Implement an evaluation.py script that uses the motmetrics library to compute MOTA and other metrics on an annotated dataset.

Record performance metrics (FPS, memory usage) on at least two videos of different lengths and contents.

Acceptance criteria: All tests pass. Evaluation script outputs metrics and shows MOTA > 0.80 with fewer than 5 ID switches per minute. Performance metrics meet the goals defined in design.md.

Task 7 – Demo and Presentation

Description: Prepare materials to demonstrate the project and summarise results.

Dependencies: Completion of Tasks 3–6.

Steps:

Use main.py to process a few videos and save the output clips to a videos/ folder.

Create a short presentation (slides) outlining the design, implementation, metrics, and sample visual results. Use images from the processed videos for illustration.

Write up README.md with instructions on installing dependencies, running the code, evaluating results, and reproducing experiments.

Acceptance criteria: Video clips clearly demonstrate tracking with IDs. Presentation and README explain the project comprehensively and highlight successes, challenges, and future work.