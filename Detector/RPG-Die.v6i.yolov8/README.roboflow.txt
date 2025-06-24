
RPG-Die - v6 2025-05-26 5:08pm
==============================

This dataset was exported via roboflow.com on May 26, 2025 at 8:08 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 1061 images.
Objects are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 608x608 (Fit (black edges))

The following augmentation was applied to create 3 versions of each source image:
* Random shear of between -5° to +5° horizontally and -5° to +5° vertically
* Random Gaussian blur of between 0 and 0.3 pixels
* Salt and pepper noise was applied to 0.3 percent of pixels


