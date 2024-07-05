import cv2
import numpy as np
import pytesseract
from datetime import datetime
import matplotlib.pyplot as plt
import re
import os
import pandas as pd

print(pytesseract)

# Configure pytesseract to use the Persian language
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
tessdata_dir_config = r'--tessdata-dir "C:\Program Files (x86)\Tesseract-OCR\tessdata" --psm 6 -l fas'


# Define traffic light color codes
RED_LIGHT = 0
YELLOW_LIGHT = 1
GREEN_LIGHT = 2


# Set traffic light timing
light_timing = {
    RED_LIGHT: 20,     # 20 seconds for red light
    YELLOW_LIGHT: 1,   # 1 second for yellow light
    GREEN_LIGHT: 3     # 3 seconds for green light
}


# Function to find rectangles that might be license plates
def find_license_plate_regions(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    _, threshold_img = cv2.threshold(abs_sobel_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    closed_img = cv2.morphologyEx(threshold_img, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    search_region_y_start = int(frame.shape[0] * 0.6)
    possible_plates = []

    for cnt in contours:
        rect = cv2.boundingRect(cnt)
        aspect_ratio = rect[2] / rect[3]
        area = cv2.contourArea(cnt)
        
        if 3.5 < aspect_ratio < 4.5 and 500 < area < 3000:
            if rect[1] + rect[3] > search_region_y_start:
                possible_plates.append(rect)
                cv2.rectangle(frame, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 255, 0), 2)
    return possible_plates


# Function to check license plate conditions
def is_valid_plate(text):
    letters_count = len(re.findall(r'[آ-ی]', text))
    numbers_count = len(re.findall(r'[۰-۹]', text))
    return letters_count <= 1 and 4 <= numbers_count <= 7


# Function to read license plate using pytesseract
def read_license_plate(frame, regions, save_path, index):
    for region in regions:
        x, y, w, h = region
        plate_img = frame[y:y+h, x:x+w]
        plate_img_path = os.path.join(save_path, f'plate_{index / 10.0}.png')
        cv2.imwrite(plate_img_path, plate_img)

        plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        _, plate_binary = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(plate_binary, config=tessdata_dir_config)
        
        if is_valid_plate(text):
            return text.strip(), plate_img_path
    return None, None


# Function to save license plate data to CSV
def save_to_csv(index, time, license_plate):
    df = pd.DataFrame([[index, time, license_plate]], columns=["Index", "Time", "License_plate"])
    df.to_csv('license_plates.csv', mode='a', header=False, index=False, encoding='utf-8')


# Function to determine traffic light status based on time
def get_traffic_light_status(elapsed_time):
    total_cycle_time = sum(light_timing.values())
    time_in_cycle = elapsed_time % total_cycle_time
    
    cumulative_time = 0
    for light_code, duration in light_timing.items():
        cumulative_time += duration
        if time_in_cycle < cumulative_time:
            return light_code
    return GREEN_LIGHT

# Function to detect crossing of yellow rectangle
def detect_yellow_rectangle_crossing(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            return True
    return False

# Function to display traffic light status
def display_traffic_light_status(frame, light_status):
    if light_status == RED_LIGHT:
        text = 'RED'
        color = (0, 0, 255)
    elif light_status == YELLOW_LIGHT:
        text = 'YELLOW'
        color = (0, 255, 255)
    else:
        text = 'GREEN'
        color = (0, 255, 0)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 4, 6)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = text_size[1] + 20
    cv2.putText(frame, text, (text_x, text_y), font, 4, color, 7)


# Initialize variables
start_time = datetime.now()
last_save_time = start_time
save_path = 'plates'
os.makedirs(save_path, exist_ok=True)
index = 1

# Create CSV file with headers
df = pd.DataFrame([["Index", "Time", "License_plate"]], columns=["Index", "Time", "License_plate"])
df.to_csv('license_plates.csv', mode='a', header=False, index=False, encoding='utf-8')

# %%
# Read video from file
cap = cv2.VideoCapture('Test_DIP_Proj.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    elapsed_time = (datetime.now() - start_time).total_seconds()
    light_status = get_traffic_light_status(elapsed_time)
    display_traffic_light_status(frame, light_status)
    
    if light_status == RED_LIGHT:
        current_time = datetime.now()
        if (current_time - last_save_time).total_seconds() >= 2:
            regions = find_license_plate_regions(frame)
            license_plate, plate_img_path = read_license_plate(frame, regions, save_path, index)
            if license_plate:
                save_to_csv(index, current_time, license_plate)
                print(f"Violation time: {current_time},\t License plate read: {license_plate}")
                index += 1
                last_save_time = current_time
    
    frame = cv2.resize(frame, (854, 480))
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Display license plate images using matplotlib
plate_images = [os.path.join(save_path, img) for img in os.listdir(save_path) if img.endswith('.png')]
plate_images = plate_images[:-1]
fig, axes = plt.subplots(1, len(plate_images), figsize=(15, 5))
for ax, img_path in zip(axes, plate_images):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    ax.axis('off')
plt.show()

# Read the CSV file containing license plate data
df = pd.read_csv('license_plates.csv')
df
