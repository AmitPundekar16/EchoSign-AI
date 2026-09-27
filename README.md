# EchoSign AI 🤟
### *Interactive Sign Language Learning Companion & Communication Assistant for Non-Verbal Individuals*

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Computer Vision](https://img.shields.io/badge/MediaPipe-Tasks%20API-orange.svg)
![Deep Learning](https://img.shields.io/badge/Model-LSTM%20%2B%20RandomForest-green.svg)
![Real-Time](https://img.shields.io/badge/Latency-%3C50ms%20Realtime-brightgreen.svg)
![Status](https://img.shields.io/badge/Status-Prototype%20MVP-success.svg)

---

## 💡 Executive Summary & Mission

Over 70 million deaf and non-verbal people worldwide rely on sign language to communicate. However, learning sign language is often an isolating journey. Traditional methods (books, static flashcards, YouTube videos) lack **real-time feedback**—a beginner cannot know whether their hand angle is correct or if they are touching the wrong area of their face.

**EchoSign AI** acts as an **interactive AI tutor and communication bridge**:
1. **Interactive Tutor Mode**: Displays guided lessons, instructs learners on hand movements, and provides instantaneous visual & auditory feedback when the sign is executed correctly.
2. **Spatial Landmark Intelligence**: Recognizes not only hand poses, but also **the spatial distance between hands and facial reference points** (e.g., distinguishing between a thumb touching the forehead for *Father* versus touching the chin for *Mother*).
3. **Temporal Motion Modeling**: Analyzes continuous time-series motions using LSTM recurrent neural networks rather than single frozen frames.

---

## 🎯 The System Architecture

```text
               +---------------------------------------------------+
               |             Webcam Video Stream (Live)            |
               +---------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                       PERCEPTION & NORMALIZATION LAYER                          |
|  * MediaPipe HandLandmarker: 21 3D landmarks per hand (Wrist, MCP, Tips)       |
|  * MediaPipe FaceLandmarker: 18 facial reference anchors (Forehead, Chin, Nose) |
|  * Distance Engine: Invariant relative metrics (Thumb-to-Chin, Index-to-Forehead)|
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                        TEMPORAL SEQUENCE PROCESSING                             |
|  * Fixed Sliding Buffer (60-frame resampled temporal window)                    |
|  * Zero-filling tolerance for temporary occlusions                              |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                         DEEP LEARNING CLASSIFICATION                            |
|  * Keras LSTM Recurrent Network (with fallback scikit-learn ensemble)           |
|  * Confidence Thresholding (>70%) & Rolling Majority Voting                     |
+---------------------------------------------------------------------------------+
                                         |
                     +-------------------+-------------------+
                     |                                       |
                     v                                       v
+-----------------------------------------+  +------------------------------------+
|            ECHO TUTOR ENGINE            |  |         FREE PRACTICE HUD          |
|  * Target sign challenge & step guide   |  |  * Real-time classification        |
|  * Dynamic timer & progress bar         |  |  * Hand bounding box overlay       |
|  * Streak multipliers & point scoring   |  |  * Confidence rating readout       |
|  * Celebratory audio chime on success   |  +------------------------------------+
+-----------------------------------------+
```

---

## 🌟 Key Features

* **AI-Guided Practice (`tutor_app.py`)**:
  * Step-by-step prompts teaching essential words (e.g., *Father*, *Mother*, *No*).
  * Real-time validation: Rewards successful execution with **+100 Points**, sound chimes, and streak bonuses.
  * Adaptive timer: Encourages fluid, deliberate signing within a friendly window.
* **Spatial Relationship Modeling (`utils/features.py`)**:
  * Solves the classic "identical hand shape" ambiguity by measuring Euclidean distances from hand tips to the forehead, chin, eyes, and nose.
* **Resilient Temporal Invariance**:
  * Sub-samples or pads input sequences uniformly, accommodating fast or slow signers.
* **Instant Extensibility (`collect_data.py` & `train_model.py`)**:
  * Adding new signs takes under 5 minutes: run the recorder, append new sequences, and retrain instantly.

---

## 🚀 Getting Started

### 1. Prerequisites & Virtual Environment

Clone the repository and set up a clean Python 3.10+ virtual environment:

```bash
# Clone repository
git clone https://github.com/AmitPundekar16/EchoSign-AI.git
cd EchoSign-AI

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # On Windows
# source venv/bin/activate  # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Interactive Tutor

Start the AI learning companion:

```bash
python tutor_app.py
```

* **Controls**:
  * `N`: Skip to next lesson
  * `M`: Toggle between **Interactive Tutor Mode** and **Free Practice Mode**
  * `R`: Reset score and streak
  * `Q`: Exit application

### 3. Run Standard Real-Time Recognition

```bash
python test_model.py
```

---

## 🧠 Extensibility: Adding Custom Signs in 3 Steps

1. **Collect Sign Data**:
   ```bash
   python collect_data.py
   # Enter sign name (e.g., WATER, HELP, THANK_YOU)
   # Press 1 to record positive examples; Press 2 for negative baseline
   ```
2. **Retrain the Network**:
   ```bash
   python train_model.py
   ```
3. **Enjoy the New Sign**: The updated model will automatically integrate into both `tutor_app.py` and `test_model.py`.

---

## 👥 Built for Impact

EchoSign AI demonstrates how lightweight, client-side computer vision can remove barriers for individuals with speech and hearing differences. By shifting focus from passive translation to active, guided learning, we empower families, caregivers, and non-verbal learners with immediate pedagogical feedback.
