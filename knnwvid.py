import tkinter as tk
from tkinter import Tk, filedialog
import cv2
from PIL import Image, ImageTk
import torch
from torchvision.transforms import functional as F
import joblib
import numpy as np

# Load YOLOv7 model
model = torch.hub.load('WongKinYiu/yolov7', 'custom', './best.pt')

# Load the saved KNN model
knn_model = joblib.load('finalized_model.sav')

def extract_color_histogram(image, bins=(8, 8, 8)):
    # Extract a 3D color histogram from the HSV color space using
    # the supplied number of `bins` per channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    # Normalize the histogram
    cv2.normalize(hist, hist)
    # Return the flattened histogram as the feature vector
    return hist.flatten()

def web_cam_func():
    def show_frame():
        _, frame = cap.read()
        
        # Perform inference
        results = model(frame)

        # Parse results and draw bounding boxes
        for *xyxy, conf, cls in results.xyxy[0]:
            if conf > 0.8:
                label = f'{model.names[int(cls)]} {conf:.2f}'
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255,0,0), 2)
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

                # Extract region of interest (ROI) using bounding box coordinates
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                roi = frame[y1:y2, x1:x2]

                # Extract features from the ROI
                roi_features = extract_color_histogram(roi).reshape(1, -1)

                # Predict the label using KNN model
                predicted_label = knn_model.predict(roi_features)

                # Display the predicted label
                cv2.putText(frame, predicted_label[0], (int(xyxy[0]), int(xyxy[1])-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Display the processed frame
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)

        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        
        lmain.after(10, show_frame)

    # Create video capture object
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 700)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)

    # Create tkinter window
    root = Tk()
    root.bind("<Escape>", lambda e: root.quit())
    root.attributes('-fullscreen', True)

    # Create main frame
    main_frame = tk.Frame(root, bg="orange")
    main_frame.place(relx=0.5, rely=0.5, width=500, height=500, anchor=tk.CENTER)

    # Create label for video display
    lmain = tk.Label(main_frame)
    lmain.place(x=0, y=0, width=500, height=500)

    # Show frames
    show_frame()

    # Execute tkinter
    root.mainloop()

web_cam_func()
