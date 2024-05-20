import tkinter as tk
from tkinter import Tk
from PIL import Image, ImageTk
import torch
import joblib
import pytesseract
from picamera2 import Picamera2
import cv2
import numpy as np

# Load YOLOv7 model
model = torch.hub.load('WongKinYiu/yolov7', 'custom', './best.pt')

# Load the saved KNN model
knn_model = joblib.load('finalized_model.sav')

def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def perform_ocr(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    license_plate_text = pytesseract.image_to_string(gray_image, config='--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    return license_plate_text

def web_cam_func():
    def show_frame():
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        results = model(frame)

        for *xyxy, conf, cls in results.xyxy[0]:
            if conf > 0.8:
                label = f'{model.names[int(cls)]} {conf:.2f}'
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255,0,0), 2)
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                roi = frame[y1:y2, x1:x2]

                roi_features = extract_color_histogram(roi).reshape(1, -1)

                predicted_label = knn_model.predict(roi_features)

                license_plate_text = perform_ocr(roi)

                cv2.putText(frame, f'Label: {predicted_label[0]}', (int(xyxy[0]), int(xyxy[1])-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv2.putText(frame, f'OCR: {license_plate_text}', (int(xyxy[0]), int(xyxy[1])-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)

        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        
        lmain.after(10, show_frame)

    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
    picam2.start()

    root = Tk()
    root.bind("<Escape>", lambda e: root.quit())
    root.attributes('-fullscreen', True)

    main_frame = tk.Frame(root, bg="orange")
    main_frame.place(relx=0.5, rely=0.5, width=500, height=500, anchor=tk.CENTER)

    lmain = tk.Label(main_frame)
    lmain.place(x=0, y=0, width=500, height=500)

    show_frame()

    root.mainloop()

web_cam_func()