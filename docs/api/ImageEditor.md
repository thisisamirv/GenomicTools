# ImageEditor

## Description
Perform image editing operations including concatenation and cropping for scientific figures, plots, and visualization materials using OpenCV-based image processing.

## Arguments
| Argument | Description |
|----------|------------|
| `-op, --operation` | Type of image operation to perform |
| `-a, --image1` | Path to the first image (for concatenation) |
| `-b, --image2` | Path to the second image (for concatenation) |
| `-c, --concat` | Direction for image concatenation |
| `-i, --image` | Path to input image (for cropping) |
| `-t, --cut` | Cut direction for cropping operation |
| `-d, --direction` | Direction of cropping within cut type |
| `-p, --percent` | Percentage of image to keep after cropping |
| `-o, --output` | Path to save the processed output image |

## Options

### `-op, --operation`
Type of image editing operation to perform:
- concatenate: Combine two images horizontally or vertically
- crop: Remove portions of an image from specified directions

### `-a, --image1`
Path to the first image file (required for concatenation).

### `-b, --image2`
Path to the second image file (required for concatenation).

### `-c, --concat`
Direction for image concatenation:
- horizontal: Place images side by side (default)
- vertical: Stack images top to bottom

### `-i, --image`
Path to input image file for cropping (required for cropping).

### `-t, --cut`
Cut direction for cropping operation:
- vertical: Cut along vertical axis
- horizontal: Cut along horizontal axis

### `-d, --direction`
Direction of cropping within the specified cut type:
- For vertical: left_to_right (default), right_to_left
- For horizontal: top_to_bottom, bottom_to_top

### `-p, --percent`
Percentage of the original image to keep:
- Range: 0.0 to 1.0
- Default: 0.5

## Usage

```sh
# Basic horizontal concatenation
GT-ImageEditor -op concatenate -a plot1.png -b plot2.png -o combined.png

# Vertical concatenation
GT-ImageEditor -op concatenate -a top_plot.png -b bottom_plot.png -c vertical -o stacked.png

# Crop left 60% of image
GT-ImageEditor -op crop -i plot.png -t vertical -d left_to_right -p 0.6 -o left_portion.png

# Crop right 40% of image
GT-ImageEditor -op crop -i plot.png -t vertical -d right_to_left -p 0.4 -o right_portion.png

# Crop top 70% of image
GT-ImageEditor -op crop -i plot.png -t horizontal -d top_to_bottom -p 0.7 -o top_portion.png
```
