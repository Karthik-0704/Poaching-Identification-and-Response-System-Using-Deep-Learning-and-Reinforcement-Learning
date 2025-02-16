import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import torch
import time

# Set Streamlit page configuration
st.set_page_config(
    page_title="YOLO Elephant Detector",
    page_icon="🐘",
    layout="centered",
    initial_sidebar_state="auto",
)

# Title and Description
st.title("🐘 YOLO Elephant Detector")
st.write("""
Upload an image, and the model will detect elephants, drawing bounding boxes around them.
""")

# Sidebar for additional options
st.sidebar.header("Configuration")

# Path to your trained YOLO model
MODEL_PATH = "runs/detect/train7/weights/best.pt"

# Device selection
device = st.sidebar.selectbox("Select Inference Device", ["cuda", "cpu"], index=0 if torch.cuda.is_available() else 1)

# Confidence slider
min_conf = st.sidebar.slider("Minimum Confidence Score", min_value=0.0, max_value=1.0, value=0.25, step=0.01)

# Warning Display
st.sidebar.info("⚠️ If you encounter warnings, they will be displayed here.")

# Check if the model exists
if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at `{MODEL_PATH}`. Please ensure the model path is correct.")
    st.stop()

@st.cache_resource
def load_model(model_path, device):
    """
    Load the YOLO model.
    Cached to prevent reloading on every interaction.
    """
    model = YOLO(model_path)
    model.to(device)
    return model

model = load_model(MODEL_PATH, device)

def annotate_image(image, detections, min_conf):
    """
    Draw bounding boxes and labels on the image for detections above the confidence threshold.
    """
    count = 0  # Counter for detected elephants
    for det in detections:
        if det.boxes is None:
            continue
        for box in det.boxes:
            confidence = box.conf[0].cpu().numpy()
            if confidence < min_conf:
                continue  # Skip boxes below the confidence threshold
            count += 1
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            label = "elephant"  # Assuming single class
            
            # Draw rectangle
            cv2.rectangle(image, (int(x1), int(y1)), 
                                 (int(x2), int(y2)), 
                                 (0, 255, 0), 2)
            
            # Put label
            cv2.putText(image, f"{label} {confidence:.2f}", 
                        (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    return image, count

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)  # Updated parameter
    
    # Convert PIL Image to OpenCV format
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Perform inference with adjusted thresholds
    with st.spinner('Detecting elephants...'):
        start_time = time.time()  # Start timing
        try:
            results = model(image_cv, conf=min_conf, iou=0.45)  # Adjust thresholds dynamically
        except Exception as e:
            st.error(f"Error during inference: {e}")
            st.stop()
        end_time = time.time()  # End timing
        inference_time = end_time - start_time  # Calculate inference time
    
    # Annotate image with detections and get total count
    annotated_image, total_count = annotate_image(image_cv.copy(), results, min_conf)
    
    # Convert back to RGB for displaying in Streamlit
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(annotated_image)
    
    # Display annotated image
    st.image(annotated_pil, caption='Detected Elephants', use_container_width=True)  # Updated parameter
    
    # Display total number of elephants detected
    st.subheader(f"Total Elephants Detected: {total_count}")
    
    # Display inference details
    st.write(f"Inference Time: {inference_time:.2f} seconds")  # Display inference time
    st.subheader("Detection Details")
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            st.write("No detections.")
            continue
        for box in boxes:
            confidence = box.conf[0].cpu().numpy()
            if confidence < min_conf:
                continue  # Skip detections below the confidence threshold
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            st.write(f"**Class:** elephant | **Confidence:** {confidence:.2f} | **Box:** [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
    
    # Provide a download link for the annotated image
    annotated_bytes = cv2.imencode('.jpg', annotated_image)[1].tobytes()
    st.download_button(
        label="Download Annotated Image",
        data=annotated_bytes,
        file_name='annotated_image.jpg',
        mime='image/jpeg'
    )
