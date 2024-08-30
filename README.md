# Face Recognition using YoloV8 and FaceNet

This repository contains code for a face recognition system using YoloV8 for face detection and FaceNet for face recognition. YoloV8 efficiently detects faces in images, while FaceNet accurately matches and recognizes the detected faces by generating unique embeddings.

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)



## Description

This project integrates YoloV8, a state-of-the-art object detection model, with FaceNet, a robust face recognition model. The system first uses YoloV8 to detect faces in an image or video. Once faces are detected, FaceNet generates embeddings for each face, which are then used to recognize and match faces against a database of known faces. This approach combines the speed of YoloV8 with the high accuracy of FaceNet, making it suitable for real-time face recognition applications.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/todap/Face-Recognition-using-YoloV8-and-FaceNet.git
    cd Face-Recognition-using-YoloV8-and-FaceNet
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Generate Face Embeddings:**
    ```bash
    python generate_face_embeddings.py
    ```

2. **Run Face Recognition:**
    ```bash
    python face_recognition.py
    ```
## Note    
Dont forget to add your paths to directory




