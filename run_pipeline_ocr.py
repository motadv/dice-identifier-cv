import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import easyocr

### --- Sub-Functions for Image Processing --- ###

def binarize_image(roi_image):
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
    binary_image = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        5, # A slightly larger block size can sometimes help
        -4
    )
    return gray, binary_image

def find_and_orient_numbers(binary_image, original_roi):
    """
    Uses EasyOCR to find numbers (1-20) in a binarized image at different rotations.
    It identifies the number with the highest confidence score from all rotations to ensure accuracy.

    Args:
        binary_image (np.ndarray): The binarized image to search for numbers.
        original_roi (PIL.Image.Image): The original cropped color image for drawing.

    Returns:
        tuple[str, np.ndarray]: The detected top-most number and the image with the number highlighted.
    """
    # Initialize EasyOCR with an allowlist to only recognize digits.
    # This improves accuracy and speed.
    reader = easyocr.Reader(['en'], gpu=True) # Set gpu=True to use GPU
    
    found_numbers = []
    (h, w) = binary_image.shape[:2]
    center = (w // 2, h // 2)

    # Rotate the image and perform OCR at each 90-degree increment
    for angle in range(0, 360, 90):
        print(f"    Processing rotation: {angle} degrees")
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(binary_image, M, (w, h),
                                       flags=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_REPLICATE)

        # Use EasyOCR to find text, restricting it to our known characters
        results = reader.readtext(rotated_image, allowlist='0123456789')

        for (bbox, text, prob) in results:
            # Further filter to ensure the number is within our expected range (1-20)
            if text.isdigit() and 1 <= int(text) <= 20:
                
                # --- Coordinate Transformation ---
                M_inv = cv2.getRotationMatrix2D(center, -angle, 1.0)
                
                original_bbox_points = []
                for point in bbox:
                    vec = np.array([point[0], point[1], 1])
                    transformed_point = np.dot(M_inv, vec)
                    original_bbox_points.append(tuple(transformed_point))

                found_numbers.append({
                    'number': text,
                    'original_bbox': np.array(original_bbox_points, dtype=np.int32),
                    'confidence': prob
                })
                print(f"    Found '{text}' (conf: {prob:.2f}) at {angle}°")

    if not found_numbers:
        print("    No valid numbers (1-20) were found by OCR.")
        return None, np.array(original_roi)

    # --- SELECTION LOGIC CHANGE ---
    # Instead of finding the top-most number, find the number with the highest confidence.
    # This avoids confusion between similar-looking numbers at different rotations (e.g., 5 vs upside-down 3).
    best_detection = max(found_numbers, key=lambda x: x['confidence'])
    
    top_number = best_detection['number']
    top_bbox_points = best_detection['original_bbox']
    top_confidence = best_detection['confidence']
    print(f"\nBest detection is: '{top_number}' with a confidence of {top_confidence:.2f}")

    # Draw the transformed bounding box on the original ROI
    output_image = cv2.cvtColor(np.array(original_roi), cv2.COLOR_RGB2BGR)
    
    # cv2.polylines can draw a polygon from a list of points
    cv2.polylines(output_image, [top_bbox_points], isClosed=True, color=(0, 255, 0), thickness=2)
    
    # Put the text label near the top-left corner of the bounding box
    label_pos = (top_bbox_points[0][0], top_bbox_points[0][1] - 10)
    cv2.putText(output_image, top_number, label_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return top_number, output_image


### --- Main Execution Logic (Unchanged) --- ###

def run_pipeline(model_path, source_path, conf_threshold, output_dir, debug):
    """
    Main pipeline function to detect objects and apply OCR.
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
            roi.save(os.path.join(instance_dir, "01_roi.jpg"))

        gray_image, binary_image = binarize_image(roi)
        if debug:
            cv2.imwrite(os.path.join(instance_dir, "02_binary_for_ocr.jpg"), binary_image)
            cv2.imwrite(os.path.join(instance_dir, "03_grayscale.jpg"), gray_image)

        # print(f"--- Processing '{instance_name}' with OCR ---")
        # detected_number, final_image = find_and_orient_numbers(binary_image, roi)

        # if detected_number:
            # final_path = os.path.join(instance_dir, f"{instance_name}_ocr_result_{detected_number}.jpg")
            # cv2.imwrite(final_path, final_image)
            # print(f"SUCCESS: Final image with OCR result saved to: {final_path}\n")
        # else:
            # print(f"FAILURE: Could not determine the top number for '{instance_name}'.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full pipeline to detect objects and find numbers using OCR.")
    parser.add_argument("--model", required=True, help="Path to the YOLOv8 model file.")
    parser.add_argument("--source", required=True, help="Path to the source image.")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold for detections.")
    parser.add_argument("--output_dir", default="pipeline_output", help="Main directory to save all outputs.")
    parser.add_argument("--debug", action="store_true", help="If set, saves all intermediate processing steps.")

    args = parser.parse_args()

    run_pipeline(args.model, args.source, args.confidence, args.output_dir, args.debug)
