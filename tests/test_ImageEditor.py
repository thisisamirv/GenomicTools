#!/usr/bin/env python
import cv2
import numpy as np
import os
import pytest
import shutil
import tempfile
from ImageEditor import ImageEditor
from utils.LoggingUtils import log

log.setup(level="DEBUG")


@pytest.fixture
def test_images_dir():
    temp_dir = tempfile.mkdtemp()

    red_square = np.zeros((100, 100, 3), dtype=np.uint8)
    red_square[:, :, 2] = 255
    cv2.imwrite(os.path.join(temp_dir, "red_square.jpg"), red_square)

    blue_rect = np.zeros((100, 200, 3), dtype=np.uint8)
    blue_rect[:, :, 0] = 255
    cv2.imwrite(os.path.join(temp_dir, "blue_rect.jpg"), blue_rect)

    green_tall = np.zeros((200, 100, 3), dtype=np.uint8)
    green_tall[:, :, 1] = 255
    cv2.imwrite(os.path.join(temp_dir, "green_tall.jpg"), green_tall)

    gradient = np.zeros((200, 200, 3), dtype=np.uint8)
    for i in range(200):
        gradient[:, i, :] = [i, i, i]
    cv2.imwrite(os.path.join(temp_dir, "gradient.jpg"), gradient)

    yield temp_dir

    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_image_paths(test_images_dir):
    return {
        "red_square": os.path.join(test_images_dir, "red_square.jpg"),
        "blue_rect": os.path.join(test_images_dir, "blue_rect.jpg"),
        "green_tall": os.path.join(test_images_dir, "green_tall.jpg"),
        "gradient": os.path.join(test_images_dir, "gradient.jpg"),
    }


@pytest.mark.unit
def test_imageeditor_initialization():
    editor = ImageEditor(
        operation="concatenate",
        output_path="test_output.jpg",
        image1="image1.jpg",
        image2="image2.jpg",
    )

    assert editor.operation == "concatenate"
    assert editor.output_path == "test_output.jpg"
    assert editor.kwargs.get("image1") == "image1.jpg"
    assert editor.kwargs.get("image2") == "image2.jpg"


@pytest.mark.unit
def test_imageeditor_invalid_operation():
    with pytest.raises(ValueError) as excinfo:
        ImageEditor(operation="resize", output_path="test_output.jpg")

    assert "Invalid operation" in str(excinfo.value)


@pytest.mark.unit
def test_imageeditor_kwargs_storage():
    editor = ImageEditor(
        operation="crop",
        output_path="test_output.jpg",
        image="test_image.jpg",
        cut="horizontal",
        direction="top_to_bottom",
        percent=0.3,
    )

    assert editor.kwargs.get("image") == "test_image.jpg"
    assert editor.kwargs.get("cut") == "horizontal"
    assert editor.kwargs.get("direction") == "top_to_bottom"
    assert editor.kwargs.get("percent") == 0.3


@pytest.mark.unit
def test_missing_required_parameters():
    editor = ImageEditor(
        operation="concatenate", output_path="test_output.jpg", image2="image2.jpg"
    )

    with pytest.raises(ValueError) as excinfo:
        editor.run()

    assert "Both image1 and image2 are required" in str(excinfo.value)

    editor = ImageEditor(operation="crop", output_path="test_output.jpg")

    with pytest.raises(ValueError) as excinfo:
        editor.run()

    assert "image is required for cropping" in str(excinfo.value)


@pytest.mark.unit
def test_concatenation_parameter_validation_with_valid_images(sample_image_paths):
    editor = ImageEditor(
        operation="concatenate",
        output_path="test_output.jpg",
        image1=sample_image_paths["red_square"],
        image2=sample_image_paths["blue_rect"],
        concat="invalid_type",
    )

    with pytest.raises(ValueError) as excinfo:
        editor.run()

    assert "Invalid concatenation type" in str(excinfo.value)


@pytest.mark.unit
def test_crop_parameter_validation_with_valid_image(sample_image_paths):
    editor = ImageEditor(
        operation="crop",
        output_path="test_output.jpg",
        image=sample_image_paths["gradient"],
        cut="diagonal",
    )

    with pytest.raises(ValueError) as excinfo:
        editor.run()

    assert "Invalid cut type" in str(excinfo.value)

    editor = ImageEditor(
        operation="crop",
        output_path="test_output.jpg",
        image=sample_image_paths["gradient"],
        cut="vertical",
        direction="inside_out",
    )

    with pytest.raises(ValueError) as excinfo:
        editor.run()

    assert "Invalid direction for vertical cut" in str(excinfo.value)

    editor = ImageEditor(
        operation="crop",
        output_path="test_output.jpg",
        image=sample_image_paths["gradient"],
        cut="horizontal",
        direction="inside_out",
    )

    with pytest.raises(ValueError) as excinfo:
        editor.run()

    assert "Invalid direction for horizontal cut" in str(excinfo.value)

    editor = ImageEditor(
        operation="crop",
        output_path="test_output.jpg",
        image=sample_image_paths["gradient"],
        percent=1.5,
    )

    with pytest.raises(ValueError) as excinfo:
        editor.run()

    assert "Percent must be between 0.0 and 1.0" in str(excinfo.value)


@pytest.mark.unit
def test_nonexistent_image_handling(output_dir):
    editor = ImageEditor(
        operation="crop",
        output_path=os.path.join(output_dir, "nonexistent.jpg"),
        image="does_not_exist.jpg",
    )

    with pytest.raises(ValueError) as excinfo:
        editor.run()

    assert "Image could not be read from" in str(excinfo.value)


@pytest.mark.unit
def test_nonexistent_images_concatenation(output_dir):
    editor = ImageEditor(
        operation="concatenate",
        output_path=os.path.join(output_dir, "nonexistent.jpg"),
        image1="does_not_exist1.jpg",
        image2="does_not_exist2.jpg",
    )

    with pytest.raises(ValueError) as excinfo:
        editor.run()

    assert "One or both images could not be read" in str(excinfo.value)


@pytest.mark.integration
def test_horizontal_concatenation_integration(sample_image_paths, output_dir):
    output_path = os.path.join(output_dir, "horizontal_concat.jpg")

    editor = ImageEditor(
        operation="concatenate",
        output_path=output_path,
        image1=sample_image_paths["red_square"],
        image2=sample_image_paths["blue_rect"],
        concat="horizontal",
    )

    editor.run()

    assert os.path.exists(output_path), "Output file was not created"

    output_img = cv2.imread(output_path)
    assert output_img.shape[0] == 100, "Height should be 100"
    assert output_img.shape[1] > 100, "Width should be larger than original red square"


@pytest.mark.integration
def test_vertical_concatenation_integration(sample_image_paths, output_dir):
    output_path = os.path.join(output_dir, "vertical_concat.jpg")

    editor = ImageEditor(
        operation="concatenate",
        output_path=output_path,
        image1=sample_image_paths["red_square"],
        image2=sample_image_paths["green_tall"],
        concat="vertical",
    )

    editor.run()

    assert os.path.exists(output_path), "Output file was not created"

    output_img = cv2.imread(output_path)
    assert output_img.shape[1] == 100, "Width should be 100"
    assert output_img.shape[0] > 100, "Height should be larger than original red square"


@pytest.mark.integration
def test_self_concatenation_integration(sample_image_paths, output_dir):
    output_path = os.path.join(output_dir, "self_concat.jpg")

    editor = ImageEditor(
        operation="concatenate",
        output_path=output_path,
        image1=sample_image_paths["red_square"],
        image2=sample_image_paths["red_square"],
    )

    editor.run()

    assert os.path.exists(output_path), "Output file was not created"

    output_img = cv2.imread(output_path)
    assert output_img.shape[0] == 100, "Height should be preserved"
    assert output_img.shape[1] == 200, "Width should be doubled"


@pytest.mark.integration
def test_different_sizes_concatenation_integration(sample_image_paths, output_dir):
    output_path = os.path.join(output_dir, "diff_size_concat.jpg")

    editor = ImageEditor(
        operation="concatenate",
        output_path=output_path,
        image1=sample_image_paths["red_square"],
        image2=sample_image_paths["gradient"],
    )

    editor.run()

    assert os.path.exists(output_path), "Output file was not created"

    output_img = cv2.imread(output_path)
    assert output_img.shape[0] == 100, "Height should match the first image"
    assert output_img.shape[1] > 100, "Width should be larger than the first image"


@pytest.mark.integration
def test_vertical_crop_left_to_right_integration(sample_image_paths, output_dir):
    output_path = os.path.join(output_dir, "vertical_left_crop.jpg")

    editor = ImageEditor(
        operation="crop",
        output_path=output_path,
        image=sample_image_paths["gradient"],
        cut="vertical",
        direction="left_to_right",
        percent=0.5,
    )

    editor.run()

    assert os.path.exists(output_path), "Output file was not created"

    output_img = cv2.imread(output_path)
    assert output_img.shape[0] == 200, "Height should be preserved"
    assert output_img.shape[1] == 100, "Width should be half of original"


@pytest.mark.integration
def test_vertical_crop_right_to_left_integration(sample_image_paths, output_dir):
    output_path = os.path.join(output_dir, "vertical_right_crop.jpg")

    editor = ImageEditor(
        operation="crop",
        output_path=output_path,
        image=sample_image_paths["gradient"],
        cut="vertical",
        direction="right_to_left",
        percent=0.5,
    )

    editor.run()

    assert os.path.exists(output_path), "Output file was not created"

    output_img = cv2.imread(output_path)
    assert output_img.shape[0] == 200, "Height should be preserved"
    assert output_img.shape[1] == 100, "Width should be half of original"

    assert np.mean(output_img) > 127, "Right half should be brighter than average"


@pytest.mark.integration
def test_horizontal_crop_top_to_bottom_integration(sample_image_paths, output_dir):
    output_path = os.path.join(output_dir, "horizontal_top_crop.jpg")

    editor = ImageEditor(
        operation="crop",
        output_path=output_path,
        image=sample_image_paths["gradient"],
        cut="horizontal",
        direction="top_to_bottom",
        percent=0.5,
    )

    editor.run()

    assert os.path.exists(output_path), "Output file was not created"

    output_img = cv2.imread(output_path)
    assert output_img.shape[0] == 100, "Height should be half of original"
    assert output_img.shape[1] == 200, "Width should be preserved"


@pytest.mark.integration
def test_horizontal_crop_bottom_to_top_integration(sample_image_paths, output_dir):
    output_path = os.path.join(output_dir, "horizontal_bottom_crop.jpg")

    editor = ImageEditor(
        operation="crop",
        output_path=output_path,
        image=sample_image_paths["gradient"],
        cut="horizontal",
        direction="bottom_to_top",
        percent=0.5,
    )

    editor.run()

    assert os.path.exists(output_path), "Output file was not created"

    output_img = cv2.imread(output_path)
    assert output_img.shape[0] == 100, "Height should be half of original"
    assert output_img.shape[1] == 200, "Width should be preserved"


@pytest.mark.integration
def test_extreme_crop_percentages_integration(sample_image_paths, output_dir):
    small_output = os.path.join(output_dir, "small_crop.jpg")
    editor = ImageEditor(
        operation="crop",
        output_path=small_output,
        image=sample_image_paths["gradient"],
        percent=0.1,
    )
    editor.run()

    assert os.path.exists(small_output), "Small crop output file was not created"
    small_img = cv2.imread(small_output)
    assert small_img.shape[1] == int(200 * 0.1), "Width should be 10% of original"

    large_output = os.path.join(output_dir, "large_crop.jpg")
    editor = ImageEditor(
        operation="crop",
        output_path=large_output,
        image=sample_image_paths["gradient"],
        percent=0.9,
    )
    editor.run()

    assert os.path.exists(large_output), "Large crop output file was not created"
    large_img = cv2.imread(large_output)
    assert large_img.shape[1] == int(200 * 0.9), "Width should be 90% of original"


@pytest.mark.integration
def test_full_workflow_concatenate_then_crop(sample_image_paths, output_dir):
    concat_output = os.path.join(output_dir, "workflow_concat.jpg")
    editor1 = ImageEditor(
        operation="concatenate",
        output_path=concat_output,
        image1=sample_image_paths["red_square"],
        image2=sample_image_paths["blue_rect"],
    )
    editor1.run()

    assert os.path.exists(concat_output), "Concatenation step failed"

    crop_output = os.path.join(output_dir, "workflow_crop.jpg")
    editor2 = ImageEditor(
        operation="crop", output_path=crop_output, image=concat_output, percent=0.75
    )
    editor2.run()

    assert os.path.exists(crop_output), "Crop step failed"

    final_img = cv2.imread(crop_output)
    assert final_img.shape[0] == 100, "Height should be preserved"
    assert final_img.shape[1] == int(
        300 * 0.75
    ), "Width should be 75% of concatenated image"


@pytest.mark.integration
def test_edge_case_single_pixel_dimension(test_images_dir, output_dir):
    tiny_img = np.ones((10, 10, 3), dtype=np.uint8) * 128
    tiny_path = os.path.join(test_images_dir, "tiny.jpg")
    cv2.imwrite(tiny_path, tiny_img)

    output_path = os.path.join(output_dir, "tiny_crop.jpg")
    editor = ImageEditor(
        operation="crop",
        output_path=output_path,
        image=tiny_path,
        percent=0.1,
    )

    editor.run()

    assert os.path.exists(output_path), "Tiny crop output file was not created"
    result_img = cv2.imread(output_path)
    assert result_img.shape[1] >= 1, "Width should be at least 1 pixel"
    assert result_img.shape[0] == 10, "Height should be preserved"


@pytest.mark.integration
def test_boundary_crop_percentages(sample_image_paths, output_dir):
    min_output = os.path.join(output_dir, "min_crop.jpg")
    editor = ImageEditor(
        operation="crop",
        output_path=min_output,
        image=sample_image_paths["gradient"],
        percent=0.01,
    )
    editor.run()

    assert os.path.exists(min_output), "Min crop output file was not created"
    min_img = cv2.imread(min_output)
    assert min_img.shape[1] >= 1, "Width should be at least 1 pixel"

    max_output = os.path.join(output_dir, "max_crop.jpg")
    editor = ImageEditor(
        operation="crop",
        output_path=max_output,
        image=sample_image_paths["gradient"],
        percent=1.0,
    )
    editor.run()

    assert os.path.exists(max_output), "Max crop output file was not created"
    max_img = cv2.imread(max_output)
    assert max_img.shape[1] == 200, "Width should be 100% of original (200 pixels)"
    assert max_img.shape[0] == 200, "Height should be preserved (200 pixels)"
