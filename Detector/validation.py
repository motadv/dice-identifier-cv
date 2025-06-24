from ultralytics import YOLO

model = YOLO("yolov8m_die_detector.pt")  # Load the trained model

# Validate the model
results = model.val()  # Validate the model
# Print the results
