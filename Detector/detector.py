import os
import cv2
from ultralytics import YOLO
import pandas as pd

detector = YOLO("runs/detect/v8m_dice2/weights/best.pt")  # Load the trained model


def process_folder(input_dir="input", base_output_dir="ROI"):
    """
    Process all images in the input directory and save the results in the output directory.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(base_output_dir, exist_ok=True)

    # Get all images files
    image_exts = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(image_exts)]

    # Process each image
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        print(f"Processing {image_path}...")

        output_dir = os.path.join(base_output_dir, os.path.splitext(image_file)[0])
        os.makedirs(output_dir, exist_ok=True)

        img = cv2.imread(image_path)
        if img is None:
            print(f"Error reading image {image_path}. Skipping...")
            continue

        # Perform inference
        results = detector.predict(
            source=image_path,
            save=True,
            save_txt=True,
            project=output_dir,
            iou=0.2
        )

        # Extract ROIs
        rois = []
        for i, box in enumerate(results[0].boxes):
            # Get coordinates of the bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Get class and confidence
            class_id = int(box.cls)
            class_name = detector.names[class_id]
            conf = box.conf.item()

            # Crop and save the ROI
            roi = img[y1:y2, x1:x2]
            roi_filename = f"{class_name}_{conf:.2f}_{i}.jpg"
            roi_path = os.path.join(output_dir, roi_filename)
            cv2.imwrite(roi_path, roi)

        # Save metadata
        rois.append(
            {
                "roi_file": roi_filename,
                "class": class_name,
                "confidence": conf,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )

        if rois:
            df = pd.DataFrame(rois)
            df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)


if __name__ == "__main__":
    process_folder(
        input_dir="input",  # Directory containing images to process
        base_output_dir="presentation",  # Base directory to save the results
    )
else:
    print("This script is intended to be run as a standalone program.")
