#!/usr/bin/env python
# Import required modules
import cv2
import numpy as np
from typing import Any, Dict
from utils.CLIFramework import CLIFramework, OptionConfig
from utils.LoggingUtils import log


class ImageEditor:
    def __init__(self, operation: str, output_path: str, **kwargs: Any) -> None:
        self.operation: str = operation
        self.output_path: str = output_path
        self.kwargs: Dict[str, Any] = kwargs
        if self.operation not in ["concatenate", "crop"]:
            raise ValueError(
                f"Invalid operation: {self.operation}. Use 'concatenate' or 'crop'"
            )

    def concatenate_images(
        self, image1_path: str, image2_path: str, concat: str = "horizontal"
    ) -> bool:
        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)
        if img1 is None or img2 is None:
            raise ValueError("One or both images could not be read")
        log.info(f"Image 1 dimensions: {img1.shape}")
        log.info(f"Image 2 dimensions: {img2.shape}")
        if concat == "horizontal":
            target_height = img1.shape[0]
            aspect_ratio = img2.shape[1] / img2.shape[0]
            new_width = int(target_height * aspect_ratio)
            img2_resized = cv2.resize(img2, (new_width, target_height))
            log.info(f"Resized image 2 to: {img2_resized.shape}")
            concatenated = np.hstack((img1, img2_resized))
        elif concat == "vertical":
            target_width = img1.shape[1]
            aspect_ratio = img2.shape[0] / img2.shape[1]
            new_height = int(target_width * aspect_ratio)
            img2_resized = cv2.resize(img2, (target_width, new_height))
            log.info(f"Resized image 2 to: {img2_resized.shape}")
            concatenated = np.vstack((img1, img2_resized))
        else:
            raise ValueError(
                "Invalid concatenation type: {concat}. Use 'horizontal' or 'vertical'"
            )
        if not cv2.imwrite(self.output_path, concatenated):
            raise ValueError("Failed to save the concatenated image")
        log.success(f"Concatenated image saved to: {self.output_path}")
        log.info(f"Final image dimensions: {concatenated.shape}")
        return True

    def crop_image(
        self,
        image_path: str,
        cut: str = "vertical",
        direction: str = "left_to_right",
        percent: float = 0.5,
    ) -> bool:
        if not 0.0 <= percent <= 1.0:
            raise ValueError(f"Percent must be between 0.0 and 1.0, got {percent}")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image could not be read from: {image_path}")
        log.info(f"Original image dimensions: {img.shape}")
        log.info(f"Cropping {percent * 100}% of the image")
        if cut == "vertical":
            if direction == "left_to_right":
                crop_width = int(img.shape[1] * percent)
                cropped = img[:, :crop_width]
                log.info(f"Keeping left {percent * 100}% (width: {crop_width})")
            elif direction == "right_to_left":
                crop_start = int(img.shape[1] * (1 - percent))
                cropped = img[:, crop_start:]
                log.info(
                    f"Keeping right {percent * 100}% (starting from: {crop_start})"
                )
            else:
                raise ValueError(
                    f"Invalid direction for vertical cut: {direction}. Use 'left_to_right' or 'right_to_left'"
                )
        elif cut == "horizontal":
            if direction == "top_to_bottom":
                crop_height = int(img.shape[0] * percent)
                cropped = img[:crop_height, :]
                log.info(f"Keeping top {percent * 100}% (height: {crop_height})")
            elif direction == "bottom_to_top":
                crop_start = int(img.shape[0] * (1 - percent))
                cropped = img[crop_start:, :]
                log.info(
                    f"Keeping bottom {percent * 100}% (starting from: {crop_start})"
                )
            else:
                raise ValueError(
                    f"Invalid direction for horizontal cut: {direction}. Use 'top_to_bottom' or 'bottom_to_top'"
                )
        else:
            raise ValueError(f"Invalid cut type: {cut}. Use 'vertical' or 'horizontal'")
        if not cv2.imwrite(self.output_path, cropped):
            raise ValueError("Failed to save the cropped image")
        log.success(f"Cropped image saved to: {self.output_path}")
        log.info(f"Final image dimensions: {cropped.shape}")
        return True

    def run(self) -> bool:
        if self.operation == "concatenate":
            if not self.kwargs.get("image1") or not self.kwargs.get("image2"):
                raise ValueError(
                    "Both image1 and image2 are required for concatenation"
                )
            log.info("Starting image concatenation")
            log.info(f"Image 1: {self.kwargs.get('image1')}")
            log.info(f"Image 2: {self.kwargs.get('image2')}")
            log.info(f"Output: {self.output_path}")
            log.info(f"Direction: {self.kwargs.get('concat', 'horizontal')}")
            return self.concatenate_images(
                image1_path=self.kwargs.get("image1"),
                image2_path=self.kwargs.get("image2"),
                concat=self.kwargs.get("concat", "horizontal"),
            )
        if self.operation == "crop":
            if not self.kwargs.get("image"):
                raise ValueError("image is required for cropping")
            log.info("Starting image cropping")
            log.info(f"Input image: {self.kwargs.get('image')}")
            log.info(f"Output: {self.output_path}")
            log.info(f"Cut: {self.kwargs.get('cut', 'vertical')}")
            log.info(f"Direction: {self.kwargs.get('direction', 'left_to_right')}")
            log.info(f"Percent: {self.kwargs.get('percent', 0.5)}")
            return self.crop_image(
                image_path=self.kwargs.get("image"),
                cut=self.kwargs.get("cut", "vertical"),
                direction=self.kwargs.get("direction", "left_to_right"),
                percent=self.kwargs.get("percent", 0.5),
            )
        raise ValueError(f"Unsupported operation: {self.operation}")


options = [
    OptionConfig(flags=["-op", "--operation"], type=str, required=True),
    OptionConfig(flags=["-a", "--image1"], type=str, default=None, required=False),
    OptionConfig(flags=["-b", "--image2"], type=str, default=None, required=False),
    OptionConfig(
        flags=["-c", "--concat"],
        type=str,
        default="horizontal",
        required=False,
        choices=["horizontal", "vertical"],
    ),
    OptionConfig(flags=["-i", "--image"], type=str, default=None, required=False),
    OptionConfig(
        flags=["-t", "--cut"],
        type=str,
        default="vertical",
        required=False,
        choices=["vertical", "horizontal"],
    ),
    OptionConfig(
        flags=["-d", "--direction"],
        type=str,
        default="left_to_right",
        required=False,
        choices=["left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"],
    ),
    OptionConfig(flags=["-p", "--percent"], type=float, default=0.5, required=False),
    OptionConfig(flags=["-o", "--output"], type=str, required=True),
]

if __name__ == "__main__":
    framework = CLIFramework(option_list=options, script_name="ImageEditor")
    opt = framework.run()
    editor = ImageEditor(
        operation=opt.operation,
        output_path=opt.output,
        image1=opt.image1,
        image2=opt.image2,
        concat=opt.concat,
        image=opt.image,
        cut=opt.cut,
        direction=opt.direction,
        percent=opt.percent,
    )
    editor.run()
