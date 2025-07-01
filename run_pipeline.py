import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

### --- Sub-Functions for Image Processing --- ###

def preprocess_for_blob_detection(roi_image):
    """
    Converts an image to grayscale and applies adaptive thresholding to isolate bright features.

    Args:
        roi_image (PIL.Image.Image): The cropped region of interest.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the grayscale image and the binary image.
    """
    # Convert PIL Image to OpenCV format (BGR) and then to grayscale
    open_cv_image = cv2.cvtColor(np.array(roi_image), cv2.COLOR_RGB2BGR)
    blurred_image = cv2.GaussianBlur(open_cv_image, (5, 5), 0)
    gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)

    # Apply Adaptive Thresholding.
    # We use THRESH_BINARY because the numbers are BRIGHTER than the background.
    binary_image = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,  # This is the corrected flag for bright numbers
        5,                 # blockSize - must be an odd number
        -4                   # C - a constant to fine-tune the threshold
    )
    return gray, binary_image

def detect_blobs(binary_image):
    """
    Finds all blobs in a binary image.

    Args:
        binary_image (np.ndarray): A binary image with white features on a black background.

    Returns:
        list[cv2.KeyPoint]: A list of all detected keypoints (blobs).
    """
    # Set up the Simple Blob Detector parameters
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 15  # Adjust this value based on the size of your numbers
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False

    detector = cv2.SimpleBlobDetector_create(params)

    # Detect all blobs
    keypoints = detector.detect(binary_image)

    if keypoints:
        print(f"INFO: Found {len(keypoints)} blob(s).")
    else:
        print("INFO: No blobs were found.")

    return keypoints

def draw_blobs_on_image(roi_image, keypoints):
    """
    Draws circles around all detected blobs on the original color image.

    Args:
        roi_image (PIL.Image.Image): The original cropped region of interest.
        keypoints (list[cv2.KeyPoint]): The list of blobs to draw.

    Returns:
        np.ndarray: The original ROI with all blobs highlighted.
    """
    # Convert original PIL ROI to OpenCV format for drawing
    output_image = cv2.cvtColor(np.array(roi_image), cv2.COLOR_RGB2BGR)

    # Draw all keypoints as rich keypoints (circles with size) in red
    if keypoints:
        output_image = cv2.drawKeypoints(
            output_image, keypoints, np.array([]), (127, 0, 0),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
    return output_image

### --- Orchestrator Function --- ###

def detect_and_draw_blobs(roi_image, instance_dir, debug=False):
    """
    Orchestrates the pipeline: preprocess, detect all blobs, and draw results.
    """
    # Step 1: Preprocess the image to get a binary version for detection
    gray_image, binary_image = preprocess_for_blob_detection(roi_image)

    # Step 2: Find all blobs in the binary image
    all_blobs = detect_blobs(binary_image)

    # Step 3: Draw all detected blobs on the original color image
    final_image = draw_blobs_on_image(roi_image, all_blobs)

    # Handle debug saving
    if debug:
        cv2.imwrite(os.path.join(instance_dir, "02_grayscale.jpg"), gray_image)
        cv2.imwrite(os.path.join(instance_dir, "03_binary_for_blob.jpg"), binary_image)

    return final_image

### --- Main Execution Logic (Unchanged) --- ###

def run_pipeline(model_path, source_path, conf_threshold, output_dir, debug):
    """
    Main pipeline function to detect objects and apply blob detection.
    """
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)
    source_img = Image.open(source_path).convert("RGB")

    results = model.predict(source=source_path, conf=conf_threshold)
    label_counts = {}

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        label = model.names[class_id]
        coords = [int(x) for x in box.xyxy[0]]

        label_counts[label] = label_counts.get(label, 0) + 1
        instance_name = f"{label}_{label_counts[label]}"

        instance_dir = os.path.join(output_dir, instance_name)
        os.makedirs(instance_dir, exist_ok=True)

        roi = source_img.crop(coords)

        if debug:
            roi_path = os.path.join(instance_dir, "01_roi.jpg")
            roi.save(roi_path)
            print(f"DEBUG: Saved initial ROI to {roi_path}")

        print(f"INFO: Processing '{instance_name}' with blob detection...")
        final_image = detect_and_draw_blobs(roi, instance_dir, debug)

        final_path = os.path.join(instance_dir, f"{instance_name}_blob_detection.jpg")
        cv2.imwrite(final_path, final_image)
        print(f"SUCCESS: Final image with blob detection saved to: {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full pipeline to detect dice and find number blobs.")
    parser.add_argument("--model", required=True, help="Path to the YOLOv8 model file.")
    parser.add_argument("--source", required=True, help="Path to the source image.")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold for detections.")
    parser.add_argument("--output_dir", default="pipeline_output", help="Main directory to save all outputs.")
    parser.add_argument("--debug", action="store_true", help="If set, saves all intermediate processing steps.")

    args = parser.parse_args()

    run_pipeline(args.model, args.source, args.confidence, args.output_dir, args.debug)