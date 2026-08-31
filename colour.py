from pathlib import Path  # Let the script resolve file paths relative to the project folder.

import time
import cv2
import matplotlib.pyplot as plt  # Plot the processed image and measurement panels for inspection.
import numpy as np  # Work with image arrays and numeric operations during the analysis.
import opencv_libraries as cv

#Colour Algorithm


AGTRON_TABLE = [  # Store the Lab reference points used to estimate the nearest Agtron value
    {"agtron": 95, "L": 83.7, "a": 0.5,  "b": 27.2},  
    {"agtron": 85, "L": 66.2, "a": 6.8,  "b": 28.5},  
    {"agtron": 75, "L": 51.2, "a": 11.1, "b": 28.0},  
    {"agtron": 65, "L": 37.1, "a": 14.3, "b": 26.9},  
    {"agtron": 55, "L": 28.5, "a": 14.0, "b": 22.0},  
    {"agtron": 45, "L": 20.9, "a": 12.4, "b": 16.4},  
    {"agtron": 35, "L": 15.8, "a": 8.9,  "b": 9.1},  
    {"agtron": 25, "L": 13.2, "a": 3.4,  "b": 0.1},  
]  
#Crop an image to an ellipse
def crop_to_ellipse(image: np.ndarray,centre: tuple[float, float],axes: tuple[float, float],angle: float = 0.0) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    if image.ndim != 3:
        raise ValueError("image must be a 3D array: (height, width, channels)")

    height, width = image.shape[:2]
    cx, cy = centre
    rx, ry = axes

    angle_rad = np.deg2rad(angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    # Bounding-box extents for a rotated ellipse
    x_extent = np.sqrt((rx * cos_angle) ** 2 + (ry * sin_angle) ** 2)
    y_extent = np.sqrt((rx * sin_angle) ** 2 + (ry * cos_angle) ** 2)

    x_min = max(0, int(np.floor(cx - x_extent)))
    x_max = min(width, int(np.ceil(cx + x_extent)) + 1)
    y_min = max(0, int(np.floor(cy - y_extent)))
    y_max = min(height, int(np.ceil(cy + y_extent)) + 1)

    # Coordinates within the cropped image
    y_coords = np.arange(y_min, y_max, dtype=np.float32)[:, None] - cy
    x_coords = np.arange(x_min, x_max, dtype=np.float32)[None, :] - cx
    #np.indices allocates two full (H, W) arrays even though x_coords only varies along 
    #columns and y_coords only varies along rows. Build them as 1D and let numpy broadcast
    #halves memory traffic, you're allocating H + W elements instead of 2*H*W

    # Rotate coordinates into the ellipse's local coordinate system
    x_rotated = cos_angle * x_coords + sin_angle * y_coords
    y_rotated = -sin_angle * x_coords + cos_angle * y_coords

    #precompute 1/rx² and 1/ry² once (scalars) and multiply as div>mul
    inv_rx2 = 1.0 / (rx * rx)
    inv_ry2 = 1.0 / (ry * ry)
    ellipse_mask = (x_rotated * x_rotated) * inv_rx2 + (y_rotated * y_rotated) * inv_ry2 <= 1.0

    cropped_image = image[y_min:y_max, x_min:x_max] * ellipse_mask[:, :, None]

    return cropped_image, ellipse_mask, (x_min, y_min)

def estimate_agtron_linear(
    measured_lab: np.ndarray,
) -> tuple[float, float]:
    measured_lab = np.asarray(measured_lab, dtype=np.float32)
    references = np.array(
        [[entry["L"], entry["a"], entry["b"]] for entry in AGTRON_TABLE],
        dtype=np.float32,)
    agtron_values = np.array(
        [entry["agtron"] for entry in AGTRON_TABLE],
        dtype=np.float32,)

    best_distance = float("inf")
    best_agtron = float(agtron_values[0])

    for index in range(len(references) - 1):
        point_a = references[index]
        point_b = references[index + 1]

        segment = point_b - point_a
        segment_length_squared = np.dot(segment, segment)

        if segment_length_squared == 0:
            t = 0.0
        else:
            # Project measured_lab onto the current line segment.
            t = np.dot(measured_lab - point_a, segment) / segment_length_squared
            t = np.clip(t, 0.0, 1.0)

        closest_point = point_a + t * segment
        distance = np.linalg.norm(measured_lab - closest_point)

        if distance < best_distance:
            best_distance = distance

            # Interpolate the Agtron value along the same segment.
            best_agtron = (
                agtron_values[index]
                + t * (agtron_values[index + 1] - agtron_values[index])
            )

    return best_agtron, best_distance

def load_image(image_path: Path) -> np.ndarray:  # Read the roast image from disk and fail clearly if it is missing.
    """Load an image and raise a useful error if loading fails."""  
    image_bgr = cv2.imread(str(image_path), 1)

    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    return image_bgr


def create_bean_mask(image_bgr: np.ndarray) -> np.ndarray:  # Mask the brown bean pixels inside the central ROI so only the bean area is analysed.
    """  
    Segment brown coffee-bean pixels inside a central elliptical region.  

    The HSV threshold values may need adjustment for different lighting  
    conditions or roast levels.  
    """  
    height, width = image_bgr.shape[:2]  
    centre = ((width // 2)+50, (height // 2)-120)  
    # Adjust proportions to fit the circular bean region
    # rn these values (0.17 and 0,25 for w and h) work for file.good.jpg ----> might need small adjustments but works well overall
    axes = (  
        int(width * 0.18),  
        int(height * 0.26),  
    )  

    cropped_image, ellipse_mask, offset = crop_to_ellipse(image=image_bgr,centre=centre,axes=axes)
    # Slight blur reduces isolated noisy pixels.
    start1 = time.perf_counter()
    blurred = cv.GaussianBlur(cropped_image, (9, 9), 0)  # Smooth the image to reduce noise before processing
    stop1 = time.perf_counter()
    print(-start1+stop1)

    # blurred = image_bgr

    # HSV separates colour from brightness more effectively than grayscale.
    # image_hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)  # Convert the image to a different colour space.
    # image_lab = cv.BGR2LAB(blurred)

    # Approximate brown coffee-bean range.
    # lower_brown = np.array([2, 35, 35], dtype=np.uint8) # for hsv
    # upper_brown = np.array([30, 255, 245], dtype=np.uint8) # for hsv
    # lower_brown = np.array([10*(255.0/100.0), -10 + 128.0, 5 + 128.0], dtype=np.uint8) #arbitrary values
    # upper_brown = np.array([75*(255.0/100.0), 40 + 128.0, 60 + 128.0], dtype=np.uint8)
    lower_brown = np.array([20, 31, 15], dtype=np.uint8) #arbitrary values
    upper_brown = np.array([73,152,255], dtype=np.uint8)

    colour_mask = cv.inRange(  # Mask only the brown coffee pixels that match the selected threshold range
        blurred,          # change this to either hsv or lab
        lower_brown,  
        upper_brown,  
    )  

    ellipse_mask_uint8 = (ellipse_mask * 255).astype(np.uint8)
    mask = np.bitwise_and(colour_mask, ellipse_mask_uint8)
    
    # # Limit processing to the central bean-containing region.
    # roi_mask = np.zeros((height, width), dtype=np.uint8)  # Initialise a blank array that will be filled with the processed output.

    # Keep only brown pixels inside the region of interest.
    # mask = np.bitwise_and(colour_mask, roi_mask)  # Combine masks to isolate only the pixels that belong to the bean or overlap region.

    return cropped_image, mask, offset


def measure_bean_colour(  # Read the Lab values of the masked bean pixels and summarise their colour as mean and median values.
    image_bgr: np.ndarray,  
    mask: np.ndarray,  
) -> tuple[np.ndarray, np.ndarray]:  
    """Return the median and mean CIELAB values of all segmented bean pixels."""  
    # image_lab_opencv = cv.BGR2LAB(image_bgr)  # Convert the image to a different colour space.

    bean_pixels_opencv = image_bgr[mask > 0]  

    if bean_pixels_opencv.size == 0:
        raise ValueError("The mask contains no bean pixels.")
    pixels = bean_pixels_opencv.astype(np.float32)  
    
    # bean_pixels_lab = np.empty_like(lab_pixels, dtype=np.float32)  
    # bean_pixels_lab[:, 0] = lab_pixels[:, 0] * 100.0 / 255.0  
    # bean_pixels_lab[:, 1] = lab_pixels[:, 1] - 128.0  
    # bean_pixels_lab[:, 2] = lab_pixels[:, 2] - 128.0  

    median = np.median(pixels, axis=0)  # Compute the median or mean Lab colour of the bean pixels so the roast colour can be summarised.
    mean = np.mean(pixels, axis=0)  # Compute the median or mean Lab colour of the bean pixels so the roast colour can be summarised.

    return median, mean


def lab_to_rgb(lab_colour: np.ndarray) -> np.ndarray:  # Convert a Lab colour back to RGB so it can be displayed as a colour patch.
    """  
    Convert one standard CIELAB colour [L*, a*, b*] to RGB.  

    The returned RGB values are floating-point values between 0 and 1,  
    which is the format expected by Matplotlib.  
    """  
    lab_pixel = np.zeros((1, 1, 3), dtype=np.float32)  # Initialise a blank array that will be filled with the processed output.

    lab_pixel[0, 0, 0] = lab_colour[0]  
    lab_pixel[0, 0, 1] = lab_colour[1]  
    lab_pixel[0, 0, 2] = lab_colour[2]  

    rgb_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_Lab2RGB)  # Convert the image to a different colour space.

    rgb_colour = np.clip(rgb_pixel[0, 0], 0.0, 1.0)  # Clamp the brightness-adjusted pixels back into the 0-255 range so the image remains valid.

    return rgb_colour


def display_results(  # Show the original image, bean mask, segmented bean view, and colour patches together for inspection.
    image_bgr,  
    mask,  
    mean_lab,  
    median_lab,  
    estimated_agtron,  
    error,  
) -> None:  
    """  
    Display the original image, mask, segmented image,  
    mean colour patch and median colour patch.  
    """  
    image_rgb = image_bgr[:, :, ::-1]  # Convert the image from BGR to RGB order without OpenCV.

    # Create the segmented image.
    segmented_rgb = np.zeros_like(image_rgb)
    segmented_rgb[mask > 0] = image_rgb[mask > 0]  

    # Convert the calculated Lab colours back to RGB for display.
    mean_rgb = lab_to_rgb(mean_lab)  
    median_rgb = lab_to_rgb(median_lab)  

    # Create solid colour patches.
    patch_height = 300
    patch_width = 300

    mean_patch = np.ones(  
        (patch_height, patch_width, 3),
        dtype=np.float32,  
    )  
    mean_patch[:] = mean_rgb  

    median_patch = np.ones(  
        (patch_height, patch_width, 3),
        dtype=np.float32,  
    )  
    median_patch[:] = median_rgb  

    # Create one figure with five images.
    figure, axes = plt.subplots(  # Create or display the plotting figure.
        1,  
        5,  
        figsize=(20, 5),  
    )  

    axes[0].imshow(image_rgb)  
    axes[0].set_title("Original image")  

    axes[1].imshow(mask, cmap="gray")  
    axes[1].set_title("Bean mask")  

    axes[2].imshow(segmented_rgb)  
    axes[2].set_title("Segmented beans")  

    axes[3].imshow(mean_patch)  
    axes[3].set_title(  
        "Mean colour\n"  
        f"L* = {mean_lab[0]:.2f}\n"
        f"a* = {mean_lab[1]:.2f}\n"
        f"b* = {mean_lab[2]:.2f}"
    )  

    axes[4].imshow(median_patch)  
    axes[4].set_title(  
    "Median Colour\n"  
    f"L* = {median_lab[0]:.2f}\n"
    f"a* = {median_lab[1]:.2f}\n"
    f"b* = {median_lab[2]:.2f}\n\n"
    f"Agtron = {estimated_agtron:.2f}\n"
    f"Distance = {error:.2f}"
    )  

    for axis in axes:  # Iterate over every item in the collection so each region is processed individually.
        axis.axis("off")  

    plt.tight_layout()  
    plt.show()  # Create or display the plotting figure.


def main() -> None: 
    start_time = time.perf_counter()
    image_path = Path("images/file.name1.jpg")  

    image_bgr = load_image(image_path)  
    cropped_image, mask, offset = create_bean_mask(image_bgr)
    end_time = time.perf_counter()

    median, mean = measure_bean_colour(cropped_image,mask)
    median_lab = cv.BGR2LABone(median)
    mean_lab = cv.BGR2LABone(mean)

    estimated_agtron, error = estimate_agtron_linear(median_lab)  
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.6f} seconds")

    display_results(  
    cropped_image,  
    mask,  
    mean_lab,  
    median_lab,  
    estimated_agtron,  
    error  
    )  


if __name__ == "__main__":  # Run the script only when this file is executed directly.
    main()  
