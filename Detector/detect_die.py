import argparse
import os
from ultralytics import YOLO
from PIL import Image

def run_extraction(model_path, source_path, conf_threshold, output_dir):
    """
    Extracts and saves regions of interest (ROIs) from an image based on YOLOv8 detections.
    """
    # --- 1. Setup ---
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the pre-trained YOLOv8 model
    model = YOLO(model_path)
    
    # Load the source image using Pillow
    source_img = Image.open(source_path)

    # --- 2. Run Inference ---
    # Run prediction on the source image
    results = model.predict(source=source_path, conf=conf_threshold)
    
    # Initialize a dictionary to count occurrences of each label for unique filenames
    label_counts = {}

    # --- 3. Process Results ---
    # The 'results' object is a list, get the first result for our single image
    result = results[0]
    
    print(f"Found {len(result.boxes)} detections in total.")

    for box in result.boxes:
        # Check if the confidence score is above the user-defined threshold
        # This check is technically redundant if 'conf' is passed to predict(),
        # but it's good practice for clarity.
        if box.conf[0] >= conf_threshold:
            # Get class name
            class_id = int(box.cls[0])
            label = model.names[class_id]
            
            # Get bounding box coordinates (left, top, right, bottom)
            coords = [int(x) for x in box.xyxy[0]]
            
            # Crop the image using the bounding box coordinates
            roi = source_img.crop(coords)
            
            # --- 4. Save the ROI ---
            # Update count for the current label to create a unique filename
            label_counts[label] = label_counts.get(label, 0) + 1
            
            # Construct the output filename
            output_filename = f"{label}_{label_counts[label]}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            
            # Convert the cropped image to RGB mode before saving as a JPEG
            rgb_roi = roi.convert("RGB")

            # Save the cropped image
            rgb_roi.save(output_path)
            print(f"Saved ROI to: {output_path}")

if __name__ == "__main__":
    # --- 5. Handle Command-Line Arguments ---
    parser = argparse.ArgumentParser(description="Extract ROIs from YOLOv8 detections.")
    parser.add_argument("--model", required=True, help="Path to the YOLOv8 model file (e.g., best.pt).")
    parser.add_argument("--source", required=True, help="Path to the source image.")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold for detections.")
    parser.add_argument("--output_dir", default="output_rois", help="Directory to save the cropped images.")
    
    args = parser.parse_args()
    
    run_extraction(args.model, args.source, args.confidence, args.output_dir)