# Traffic Red-Light Violation Detection (OCR)🚦

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-green)
![Tesseract](https://img.shields.io/badge/Tesseract-OCR-yellow)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Results](#results)
- [License](#license)

## Overview
This project demonstrates a system for detecting red light violations using computer vision and OCR (Optical Character Recognition). The system captures video frames, detects license plates, and reads the license plate numbers when a vehicle crosses a red light.

## Features
- Detects red light violations in real-time.
- Reads license plate numbers using OCR.
- Saves violation data to a CSV file.
- Displays detected license plates.


## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/4prince8/Traffic_Red_Light_Violation_Detection_OCR.git
    cd Traffic_Red_Light_Violation_Detection_OCR
    ```

2. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Install Tesseract OCR:**
    - Download and install Tesseract from [here](https://github.com/tesseract-ocr/tesseract).
    - Ensure the `tesseract.exe` path is correctly set in the script.

## Usage

1. **Run the script:**
    ```bash
    python Traffic_violation.py
    ```

2. **Press 'q' to quit the video display window.**

## Configuration

- **Traffic Light Timing:**

   You can change Timing.

    ```python
    light_timing = {
        RED_LIGHT: 20,     # 20 seconds for red light
        YELLOW_LIGHT: 1,   # 1 second for yellow light
        GREEN_LIGHT: 3     # 3 seconds for green light
    }
    ```

- **Tesseract Configuration:**

   Put your `path` in the part below.

    ```python
    pytesseract.pytesseract.tesseract_cmd = r'YOUR_PATH\Tesseract-OCR\tesseract.exe'
    tessdata_dir_config = r'--tessdata-dir "YOUR_PATH\Tesseract-OCR\tessdata" --psm 6 -l fas'
    ```

- **Video Path:**

   Put your Video Path instead of `Test_DIP_Proj.mp4`.

    ```python
    # Read video from file
    cap = cv2.VideoCapture('Test_DIP_Proj.mp4')
    ```
## Results

The detected license plates are saved as images in the `plates` directory and logged into a CSV file named `license_plates.csv`.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
