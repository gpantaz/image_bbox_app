# image_bbox_app

This is an app that utilizes Gradio to create object-level annotations for images.

## Install
`conda create -n image_app python=3.9`

`conda activate image_app`

`pip install -r requirements.txt`


## How to run
Assuming you have a list of images within your `image` folder, create a txt file containing all the object classes similar to `example_classes.txt`

Then:

`python image_annotation_app.py --input_image_directory image --input_image_classes_path example_classes.txt --output_annotation_json image_annotations.json`

This will create a local and sharable url that can be used to annotate any images within the folder.

## How to annotate

1. Select an the image you want to annotate.
   1. You can use the `previous image` & `next image` to go to the previous / next image within the folder
   2. Alternatively, you can type the name of the image you want within the `Image` textbox and press `Go to Image`
   3. You can also use the slider to jump to an image
2. Use the slider for (x_min, y_min, x_max, y_max) to create a bounding box for that image. These values can also be modified if you write within the upper right textbox of each slider panel.
3. Use the dropdown options to select a class for the above bounding box.
4. Click add bbox annotation. This will save the annotation within the `output_annotation_json` path. If you go back to this image the annotation will be loaded and shown in the Image annotation metadata panel.


## How to remove an annotation
At the moment you can remove the latest annotation for an image by just clicking the `Remove bbox annotation` button
