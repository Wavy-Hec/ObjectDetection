# INTERACTIVE TUTORIAL - Understanding Your Project

**Goal:** Understand what we built and how it all works  
**Time:** 15-20 minutes  
**Follow along:** Copy and paste each command

---

## PART 1: What You Have Now (5 minutes)

### Step 1: See Your Project Structure

```bash
cd /home/hector/LLM/ObjectDetection
ls -la
```

**What you'll see:**
```
PAD.md                      ← Big document explaining the whole system
README.md                   ← Start here guide
WEEK5_DELIVERABLE.md        ← Assignment summary
TESTING_GUIDE.md            ← How to test
NEXT_STEPS.md               ← What to do next
QUICK_REFERENCE.md          ← Quick navigation

templates/                  ← Copy these when starting new features
commands/                   ← Instructions for AI assistant
feature-1/                  ← Your first completed feature
src/                        ← Your actual code
tests/                      ← Your tests
```

**What this means:** Everything is now organized instead of messy!

---

### Step 2: Look at Your Code

```bash
# See what code you have
ls src/
```

**Output:** `detector.py`

This is your actual working code - the object detector!

```bash
# See how many lines of code
wc -l src/detector.py
```

**Output:** About 128 lines

```bash
# Look at the first 30 lines to see what it does
head -30 src/detector.py
```

**What you'll see:**
- Imports for YOLO and PyTorch
- A `Detection` class (stores found objects)
- An `ObjectDetector` class (finds people and cars in images)

---

### Step 3: Look at Your Tests

```bash
# See what tests you have
ls tests/
```

**Output:**
```
__init__.py
conftest.py
test_detector.py
```

```bash
# See how many tests
grep "def test_" tests/test_detector.py | wc -l
```

**Output:** 10 tests

```bash
# See what the tests are called
grep "def test_" tests/test_detector.py
```

**What you'll see:** Names of all 10 tests that check your code works

---

## PART 2: Understanding What Each Piece Does (10 minutes)

### The Code (src/detector.py)

**What it does:** Finds people and cars in images using AI

```bash
# Read the docstring at the top
head -10 src/detector.py
```

**How it works:**
1. You give it an image (a frame from a video)
2. It runs YOLO AI on the image
3. It returns boxes around people and cars it found

**Try it yourself:**
```bash
conda activate object_tracking
python src/detector.py
```

**What happens:**
1. Downloads AI model (first time only)
2. Creates a blank test image
3. Tries to find objects
4. Reports results

---

### The Tests (tests/test_detector.py)

**What they do:** Make sure your code works correctly

```bash
# See what one test looks like
sed -n '/def test_detection_creation/,/assert det.confidence/p' tests/test_detector.py
```

**What this test does:**
1. Creates a Detection object
2. Checks it has the right properties
3. Makes sure nothing crashes

**Types of tests we have:**
1. **Basic tests** - Does the class work?
2. **Configuration tests** - Can you change settings?
3. **Integration tests** - Does everything work together?

---

### The Templates (templates/)

**What they do:** Starting points for new features

```bash
# See what templates exist
ls templates/
```

**Output:**
```
PRD_template.md      ← Template for requirements
task_template.md     ← Template for tasks
```

```bash
# Look at what's in the PRD template
head -50 templates/PRD_template.md
```

**How you use it:**
1. Copy template: `cp templates/PRD_template.md feature-2/PRD.md`
2. Fill in the blanks
3. Now you have requirements for feature-2!

---

### The Commands (commands/)

**What they do:** Tell you (and AI) what to do at each step

```bash
# See what commands exist
ls commands/
```

**Output:**
```
design.md        ← How to design a feature
plan.md          ← How to break into tasks
implement.md     ← How to write code
validate.md      ← How to test everything
```

```bash
# Look at the design command
head -60 commands/design.md
```

**How you use it:**
1. Read the command file
2. Copy the AI prompt
3. Paste to AI assistant
4. AI creates your design document!

---

### The Documentation

```bash
# See the main documents
ls *.md
```

**Each document explained:**

**PAD.md** - Project Architecture Document
- What the whole system does
- How it's built
- Why we made certain choices

```bash
# See how big it is
wc -l PAD.md
```

**README.md** - Getting Started Guide
- How to set up
- How to use the code
- Where to find things

**TESTING_GUIDE.md** - How to Test
- Step-by-step testing instructions
- What each test does
- Troubleshooting

**WEEK5_DELIVERABLE.md** - Assignment Summary
- What you completed
- How to present it
- Questions you can answer

---

## PART 3: Actually Using It (5 minutes)

### Test 1: Run the Detector

```bash
# Make sure you're in the right environment
conda activate object_tracking

# Run the detector
python src/detector.py
```

**What you should see:**
```
ObjectDetector initialized on cpu (or cuda if you have GPU)
Model: yolov5s.pt, Confidence threshold: 0.25
Target classes: {'person', 'car'}

Test detection completed.
Number of detections: 0
Detector module test completed successfully!
```

**What this means:** ✅ Your detector works!

---

### Test 2: Run Unit Tests

```bash
# Run all tests
pytest tests/ -v
```

**What you should see:**
```
tests/test_detector.py::TestDetection::test_detection_creation PASSED
tests/test_detector.py::TestDetection::test_detection_repr PASSED
... (8 more tests)

============================== 10 passed in X.XXs ==============================
```

**What this means:** ✅ All your tests pass!

---

### Test 3: Check Code Coverage

```bash
# See how much code is tested
pytest tests/ --cov=src --cov-report=term
```

**What you should see:**
```
Name              Stmts   Miss  Cover
-------------------------------------
src/detector.py      48     17    65%
-------------------------------------
TOTAL                48     17    65%
```

**What this means:**
- 65% of your code has tests
- Goal is 80%
- Need more tests for feature-2

---

### Test 4: Use the Detector in Python

```bash
# Open Python
python
```

Then type this in Python:

```python
from src.detector import ObjectDetector
import numpy as np

# Create detector
detector = ObjectDetector()

# Create a blank image (480 pixels tall, 640 wide, 3 color channels)
blank_image = np.zeros((480, 640, 3), dtype=np.uint8)

# Try to detect objects
detections = detector.detect(blank_image)

# See results
print(f"Found {len(detections)} objects")

# Exit Python
exit()
```

**What this shows:** How to use your detector from Python code!

---

## PART 4: What To Do Next

### Option 1: Just Understand It (For Discussion)

You don't need to do anything more! You have:
- ✅ Working code
- ✅ Passing tests  
- ✅ Complete documentation
- ✅ Everything needed for Week 5

**For your discussion, just:**
1. Show the tests passing: `pytest tests/ -v`
2. Show the structure: `ls -la`
3. Explain what you learned (read `feature-1/post_development_analysis.md`)

---

### Option 2: Start Feature-2 (Later, After Discussion)

When ready to build the tracker:

```bash
# Create new branch
git checkout -b feature-2

# Create feature directory
mkdir -p feature-2/tasks

# Copy PRD template
cp templates/PRD_template.md feature-2/PRD.md

# Edit it (add requirements for tracker)
nano feature-2/PRD.md  # or your editor

# Then follow commands/design.md to create design
# Then commands/plan.md to break into tasks
# Then commands/implement.md to write code
# Then commands/validate.md to test
```

---

## PART 5: Quick Command Reference

### Daily Commands

```bash
# Activate environment
conda activate object_tracking

# Run tests
pytest tests/ -v

# Run code
python src/detector.py

# Check coverage
pytest tests/ --cov=src
```

### Git Commands

```bash
# See what changed
git status

# See recent commits
git log --oneline -5

# Create new branch
git checkout -b branch-name

# Push changes
git push origin main
```

### Navigation

```bash
# Where am I?
pwd

# List files
ls -la

# Read a file
cat filename.md

# Read first 20 lines
head -20 filename.md

# Search in files
grep "search term" filename
```

---

## PART 6: Understanding the Big Picture

### What We Built

**Before:**
```
ObjectDetection/
├── PRD.md           (messy, all in root)
├── design.md
├── plan.md
├── detector.py
└── task files...
```

**After:**
```
ObjectDetection/
├── PAD.md                    (architecture)
├── README.md                 (guide)
├── templates/                (reusable)
├── commands/                 (instructions)
├── feature-1/                (iteration 1)
│   ├── PRD.md
│   ├── design.md
│   ├── plan.md
│   └── post_development_analysis.md
├── src/                      (code)
│   └── detector.py
└── tests/                    (tests)
    └── test_detector.py
```

**Why this is better:**
1. **Organized** - Easy to find things
2. **Reusable** - Templates for next features
3. **Scalable** - Can add feature-2, feature-3, etc.
4. **Documented** - Clear what everything does
5. **Testable** - Tests verify it works

---

### The Workflow (How to Build Features)

```
For each new feature:

1. DESIGN PHASE
   ├─ Copy PRD template
   ├─ Fill in requirements
   ├─ Use commands/design.md
   └─ Create design.md

2. PLANNING PHASE
   ├─ Use commands/plan.md
   ├─ Break into tasks
   └─ Create plan.md

3. IMPLEMENTATION PHASE
   ├─ Use commands/implement.md
   ├─ Write code in src/
   ├─ Write tests in tests/
   └─ Run tests often

4. VALIDATION PHASE
   ├─ Use commands/validate.md
   ├─ Run all tests
   ├─ Check coverage
   └─ Create validation report

5. POST-DEV ANALYSIS
   ├─ Document what worked
   ├─ Document what didn't
   ├─ Update templates
   └─ Improve process
```

---

## Summary: You Now Understand

✅ **Your code** - src/detector.py finds people and cars  
✅ **Your tests** - 10 tests verify code works  
✅ **Your templates** - Starting points for new features  
✅ **Your commands** - Instructions for each phase  
✅ **Your workflow** - Repeatable process  

**You're ready to:**
- Discuss your Week 5 work
- Explain what you learned
- Start feature-2 when ready

---

## Still Confused? Start Here

1. **Read README.md** - 5 minute overview
   ```bash
   cat README.md
   ```

2. **Run the tests** - See it work
   ```bash
   conda activate object_tracking
   pytest tests/ -v
   ```

3. **Read NEXT_STEPS.md** - What to do
   ```bash
   cat NEXT_STEPS.md
   ```

4. **Ask questions!** - It's okay to not understand everything at once

---

**Created:** 2025-01-27  
**Purpose:** Help you understand your project  
**Next:** Try the commands above and see what happens!
