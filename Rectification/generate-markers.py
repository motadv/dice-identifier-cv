import cv2
import os

def generate_aruco_markers(output_dir='./markers'):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    
    # Create the markers directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i in range(4):
        marker_image = cv2.aruco.generateImageMarker(aruco_dict, i, 200)
        # Save the marker image to the specified output directory
        cv2.imwrite(f'{output_dir}/marker_{i}.png', marker_image)
    print(f"Markers generated and saved in {output_dir} as marker_0.png, marker_1.png, marker_2.png, marker_3.png")


if __name__ == "__main__":
    generate_aruco_markers()
    