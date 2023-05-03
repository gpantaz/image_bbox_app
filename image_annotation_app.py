import argparse
import glob
import json
import os
from collections.abc import Iterator
from typing import Any, Optional, Union

import cv2
import gradio as gr
import numpy as np
from numpy import typing


ImageAnnotation = dict[str, list[dict[str, Any]]]


class PlotBoundingBoxes:
    """Class for plotting bounding boxes on image.

    Adapted from https://github.com/pzzhang/VinVL.
    """

    def __init__(self, color: tuple[int, int, int]) -> None:
        self._default_font = cv2.FONT_HERSHEY_SIMPLEX
        self.color = color

    def draw_bb(
        self,
        image: typing.NDArray[np.uint8],
        boxes_coords: list[list[float]],
        boxes_labels: list[Any],
        probs: Optional[typing.NDArray[np.float32]] = None,
        draw_label: Optional[bool] = True,
    ) -> None:
        """Plot the bounding boxes."""
        self._width = image.shape[1]
        self._height = image.shape[0]
        self._font_info = self._get_font_info()

        color = self._get_color(boxes_labels)

        for idx, (rect, label) in enumerate(zip(boxes_coords, boxes_labels)):
            (start_point, end_point) = self._get_start_end_positions(rect)
            cv2.rectangle(
                image,
                start_point,
                end_point,
                color[label],
                self._font_info["font_thickness"],
            )
            if draw_label:
                rect_label = f"{label}"
                if probs is not None:
                    rect_label = f"{label}-{probs[idx]:.2f}"
                self._annotate(image, rect, rect_label, color[label])

    def _annotate(
        self,
        image: typing.NDArray[np.uint8],
        rect: list[float],
        rect_label: str,
        color_label: tuple[int, int, int],
    ) -> None:
        """Annotate a bounding box."""

        def gen_candidate() -> Iterator[tuple[int, int]]:  # noqa: WPS430
            """Get coordinates for text."""
            # above of top left
            yield int(rect[0]) + 2, int(rect[1]) - 4
            # below of bottom left
            yield int(rect[0]) + 2, int(rect[3]) + text_height + 2  # noqa: WPS221

        (_, text_height), _ = cv2.getTextSize(
            rect_label,
            self._font_info["font"],
            self._font_info["font_scale"],
            self._font_info["font_thickness"],
        )
        for text_left, text_bottom in gen_candidate():
            should_write_text = (
                0 <= text_left < self._width - 12  # noqa: WPS432
                and 12 < text_bottom < self._height  # noqa: WPS432
            )
            if should_write_text:
                self._put_text(
                    image=image,
                    text=rect_label,
                    bottomleft=(text_left, text_bottom),
                    color=color_label,
                )
                break

    def _get_start_end_positions(
        self, rect: list[float]
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Get start and end positions for a bounding box."""
        start = (int(rect[0]), int(rect[1]))
        end = (int(rect[2]), int(rect[3]))
        return (start, end)

    def _get_color(self, boxes_labels: list[Any]) -> dict[Any, tuple[int, int, int]]:
        """Get label color for a bounding box."""
        color: dict[Any, tuple[int, int, int]] = {}
        dist_label = set(boxes_labels)
        for label in dist_label:
            color[label] = self.color
        return color

    def _get_font_info(self) -> dict[str, Union[int, float]]:
        """Get font."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        ref = (self._height + self._width) / 2
        font_scale = ref / 1000
        font_thickness = int(max(ref / 400, 1))  # noqa: WPS432
        return {
            "font": font,
            "font_scale": font_scale,
            "font_thickness": font_thickness,
        }

    def _put_text(
        self,
        image: typing.NDArray[np.uint8],
        text: str,
        bottomleft: tuple[int, int] = (0, 100),
        color: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        """Write text to image."""
        cv2.putText(
            image,
            text,
            bottomleft,
            self._default_font,
            self._font_info["font_scale"],
            color,
            self._font_info["font_thickness"],
        )


class ImageVisualizer:
    """Class for visualising and annotating turns from arena sessions."""

    def __init__(
        self,
        input_image_directory: str,
        input_image_classes_path: str,
        output_annotation_json: str,
    ) -> None:
        self.input_image_directory = input_image_directory

        with open(input_image_classes_path) as fp:
            lines = fp.readlines()

        self.classes = sorted([line.strip() for line in lines])
        self.output_annotation_json = output_annotation_json

        extensions = [".png", ".jpeg"]
        images = []
        for extension in extensions:
            images.extend(
                glob.glob(os.path.join(input_image_directory, f"*{extension}"))
            )

        self.images = sorted(images)
        self._bbox_plot = PlotBoundingBoxes(color=(255, 255, 255))
        self._cache_image = "image_bbox.png"

    def __len__(self) -> int:
        """Return the number of images."""
        return len(self.images)

    def on_previous_image(self, image_index: int) -> tuple[int, str, ImageAnnotation]:
        """Get the previous image."""
        new_image_index = max(0, image_index - 1)
        image_name = os.path.basename(self.images[new_image_index])
        image_annotation: ImageAnnotation = {image_name: []}
        if os.path.exists(self.output_annotation_json):
            with open(self.output_annotation_json) as fp:
                data = json.load(fp)
                image_ann = data.get(image_name, [])
                image_annotation = {image_name: image_ann}
        return new_image_index, self.images[new_image_index], image_annotation

    def on_next_image(self, image_index: int) -> tuple[int, str, ImageAnnotation]:
        """Get the next image."""
        new_image_index = min(len(self.images) - 1, image_index + 1)
        image_name = os.path.basename(self.images[new_image_index])
        image_annotation: ImageAnnotation = {image_name: []}
        if os.path.exists(self.output_annotation_json):
            with open(self.output_annotation_json) as fp:
                data = json.load(fp)
                image_ann = data.get(image_name, [])
                image_annotation = {image_name: image_ann}
        return new_image_index, self.images[new_image_index], image_annotation

    def on_jump_image_slider(
        self, image_index: int
    ) -> tuple[str, str, ImageAnnotation]:
        """Go to an image provided by its index."""
        image_name = os.path.basename(self.images[image_index])
        image_annotation: ImageAnnotation = {image_name: []}
        if os.path.exists(self.output_annotation_json):
            with open(self.output_annotation_json) as fp:
                data = json.load(fp)
                image_ann = data.get(image_name, [])
                image_annotation = {image_name: image_ann}
        return (
            os.path.basename(self.images[image_index]),
            self.images[image_index],
            image_annotation,
        )

    def on_jump_image_textbox(
        self, image_name: str
    ) -> tuple[int, str, ImageAnnotation]:
        """Go to an image provided by its name."""
        image_index = self.images.index(
            os.path.join(self.input_image_directory, image_name)
        )
        image_name = os.path.basename(self.images[image_index])
        image_annotation: ImageAnnotation = {image_name: []}
        if os.path.exists(self.output_annotation_json):
            with open(self.output_annotation_json) as fp:
                data = json.load(fp)
                image_ann = data.get(image_name, [])
                image_annotation = {image_name: image_ann}
        return image_index, self.images[image_index], image_annotation

    def on_update_annotation(
        self,
        image_index: int,
        object_class: str,
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int,
    ) -> ImageAnnotation:
        """Update the annotation."""
        image_name = os.path.basename(self.images[image_index])
        image_annotation = []
        if os.path.exists(self.output_annotation_json):
            with open(self.output_annotation_json) as fp:
                data = json.load(fp)
            image_annotation = data.get(image_name, [])

        image_annotation.append(
            {"bbox": [x_min, y_min, x_max, y_max], "object_class": object_class}
        )
        return {image_name: image_annotation}

    def on_add_bbox_annotation(
        self,
        image_name: str,
        image_annotation: ImageAnnotation,
        object_class: str,
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int,
    ) -> ImageAnnotation:
        """Add a bounding box annotation for an image."""
        image_ann = image_annotation.get(image_name, None)
        if image_ann is not None:
            image_annotation[image_name].append(
                {"bbox": [x_min, y_min, x_max, y_max], "object_class": object_class}
            )
        else:
            image_annotation[image_name] = [
                {"bbox": [x_min, y_min, x_max, y_max], "object_class": object_class}
            ]

        self.on_save_annotation(image_annotation)
        return image_annotation

    def on_remove_bbox_annotation(
        self, image_name: str, image_annotation: ImageAnnotation
    ) -> ImageAnnotation:
        """Remove the latest bounding box annotation for an image."""
        image_ann = image_annotation.get(image_name, None)
        if image_ann is not None:
            image_annotation[image_name] = image_annotation[image_name][:-1]
            self.on_save_annotation(image_annotation)

        return image_annotation

    def on_save_annotation(self, image_annotation: ImageAnnotation) -> None:
        """Save the annotations."""
        data = {}
        if os.path.exists(self.output_annotation_json):
            with open(self.output_annotation_json) as fpr:
                data = json.load(fpr)

        data.update(image_annotation)
        with open(self.output_annotation_json, "w") as fpw:
            json.dump(data, fpw, indent=4)

    def on_coords_slider(
        self,
        image_index: int,
        object_class: str,
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int,
    ) -> tuple[typing.NDArray[np.uint8], ImageAnnotation]:
        """Draw a bounding box annotation for an image."""
        bboxes_coords = [[float(x_min), float(y_min), float(x_max), float(y_max)]]
        image = cv2.imread(self.images[image_index])
        self._bbox_plot.draw_bb(
            image=image,
            boxes_coords=bboxes_coords,
            boxes_labels=[object_class],
            draw_label=True,
        )
        annotation = self.on_update_annotation(
            image_index, object_class, x_min, y_min, x_max, y_max
        )
        return image[:, :, ::-1], annotation


def main(args: argparse.Namespace) -> None:
    """Main."""
    image_visualizer = ImageVisualizer(
        input_image_directory=args.input_image_directory,
        input_image_classes_path=args.input_image_classes_path,
        output_annotation_json=args.output_annotation_json,
    )

    with gr.Blocks() as block:
        with gr.Row():
            image_textbox = gr.Textbox(
                label="Image \U0000270D",
                interactive=True,
                value=os.path.basename(image_visualizer.images[0]),
            )

        with gr.Row():
            previous_image_button = gr.Button("Previous Image", label="Previous Image")
            next_image_button = gr.Button(
                "Next Image",
                label="Next image",
            )
            jump_image_button = gr.Button(
                "Go To Image",
                label="Go to image",
            )
        with gr.Row():
            jump_image_slider = gr.Slider(
                minimum=0,
                maximum=len(image_visualizer) - 1,
                label="Jump to image",
                value=0,
                step=1,
            )

        with gr.Row():
            output_image_gallery = gr.Image(
                label="Image",
                value=image_visualizer.images[0],
            )

        with gr.Row():
            with gr.Column():
                x_min_slider = gr.Slider(
                    label="X min",
                    interactive=True,
                    value=0,
                    minimum=0,
                    maximum=args.max_width,
                    step=1,
                )

            with gr.Column():
                y_min_slider = gr.Slider(
                    label="Y min",
                    interactive=True,
                    value=0,
                    minimum=0,
                    maximum=args.max_height,
                    step=1,
                )

            with gr.Column():
                x_max_slider = gr.Slider(
                    label="X max",
                    interactive=True,
                    value=0,
                    minimum=0,
                    maximum=args.max_width,
                    step=1,
                )

            with gr.Column():
                y_max_slider = gr.Slider(
                    label="Y max",
                    interactive=True,
                    value=0,
                    minimum=0,
                    maximum=args.max_height,
                    step=1,
                )

        with gr.Row():
            object_class_dropdown = gr.Dropdown(
                label="Object class",
                choices=image_visualizer.classes,
            )

            with gr.Column():
                image_annotation_metadata = gr.JSON(
                    label="Image annotation metadata", value={}
                )

        with gr.Row():
            remove_bbox_annotation = gr.Button(
                "Remove bbox annotation",
                label="Remove bbox annotation",
            )

            add_bbox_annotation = gr.Button(
                "Add bbox annotation",
                label="Add bbox annotation",
                variant="primary",
            )

        previous_image_button.click(
            fn=image_visualizer.on_previous_image,
            inputs=[jump_image_slider],
            outputs=[
                jump_image_slider,
                output_image_gallery,
                image_annotation_metadata,
            ],
        )

        next_image_button.click(
            fn=image_visualizer.on_next_image,
            inputs=[jump_image_slider],
            outputs=[
                jump_image_slider,
                output_image_gallery,
                image_annotation_metadata,
            ],
        )

        jump_image_button.click(
            fn=image_visualizer.on_jump_image_textbox,
            inputs=[image_textbox],
            outputs=[
                jump_image_slider,
                output_image_gallery,
                image_annotation_metadata,
            ],
        )

        jump_image_slider.change(
            fn=image_visualizer.on_jump_image_slider,
            inputs=[jump_image_slider],
            outputs=[
                image_textbox,
                output_image_gallery,
                image_annotation_metadata,
            ],
        )

        x_min_slider.change(
            fn=image_visualizer.on_coords_slider,
            inputs=[
                jump_image_slider,
                object_class_dropdown,
                x_min_slider,
                y_min_slider,
                x_max_slider,
                y_max_slider,
            ],
            outputs=[output_image_gallery, image_annotation_metadata],
        )

        y_min_slider.change(
            fn=image_visualizer.on_coords_slider,
            inputs=[
                jump_image_slider,
                object_class_dropdown,
                x_min_slider,
                y_min_slider,
                x_max_slider,
                y_max_slider,
            ],
            outputs=[output_image_gallery, image_annotation_metadata],
        )

        x_max_slider.change(
            fn=image_visualizer.on_coords_slider,
            inputs=[
                jump_image_slider,
                object_class_dropdown,
                x_min_slider,
                y_min_slider,
                x_max_slider,
                y_max_slider,
            ],
            outputs=[output_image_gallery, image_annotation_metadata],
        )

        y_max_slider.change(
            fn=image_visualizer.on_coords_slider,
            inputs=[
                jump_image_slider,
                object_class_dropdown,
                x_min_slider,
                y_min_slider,
                x_max_slider,
                y_max_slider,
            ],
            outputs=[output_image_gallery, image_annotation_metadata],
        )

        object_class_dropdown.change(
            fn=image_visualizer.on_update_annotation,
            inputs=[
                jump_image_slider,
                object_class_dropdown,
                x_min_slider,
                y_min_slider,
                x_max_slider,
                y_max_slider,
            ],
            outputs=[image_annotation_metadata],
        )

        add_bbox_annotation.click(
            fn=image_visualizer.on_save_annotation,
            inputs=[
                image_annotation_metadata,
            ],
        )

        remove_bbox_annotation.click(
            fn=image_visualizer.on_remove_bbox_annotation,
            inputs=[image_textbox, image_annotation_metadata],
            outputs=[image_annotation_metadata],
        )

        block.launch(share=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image_directory", required=True)
    parser.add_argument("--input_image_classes_path", default="example_classes.txt")
    parser.add_argument("--max_height", type=int, default=300)  # noqa: WPS432
    parser.add_argument("--max_width", type=int, default=300)  # noqa: WPS432
    parser.add_argument(
        "--output_annotation_json",
        default="image_annotations.json",
        help="Path to output annotation json file.",
    )

    args = parser.parse_args()
    main(args)
