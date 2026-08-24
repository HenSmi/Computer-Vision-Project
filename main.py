from pathlib import Path  # Let the script resolve file paths relative to the project folder.

import cv2  # Use OpenCV to read, filter, and display the roast images.
import matplotlib.pyplot as plt  # Plot the processed image and measurement panels for inspection.
import numpy as np  # Work with image arrays and numeric operations during the analysis.

#Colour Algorithm


AGTRON_TABLE = [  # Store the Lab reference points used to estimate the nearest Agtron value.
    {"agtron": 95, "L": 83.7, "a": 0.5,  "b": 27.2},  
    {"agtron": 85, "L": 66.2, "a": 6.8,  "b": 28.5},  
    {"agtron": 75, "L": 51.2, "a": 11.1, "b": 28.0},  
    {"agtron": 65, "L": 37.1, "a": 14.3, "b": 26.9},  
    {"agtron": 55, "L": 28.5, "a": 14.0, "b": 22.0},  
    {"agtron": 45, "L": 20.9, "a": 12.4, "b": 16.4},  
    {"agtron": 35, "L": 15.8, "a": 8.9,  "b": 9.1},  
    {"agtron": 25, "L": 13.2, "a": 3.4,  "b": 0.1},  
]  

def estimate_agtron(measured_lab):  # Compare the measured bean colour to the Agtron reference table and return the closest match.
    """  
    Estimate the Agtron value by finding the nearest reference  
    in CIELAB space.  
    """  

    best_agtron = None  
    smallest_distance = float("inf")  

    for sample in AGTRON_TABLE:  # Iterate over every item in the collection so each region is processed individually.

        reference = np.array([
            sample["L"],  
            sample["a"],  
            sample["b"]  
        ])  

        distance = np.linalg.norm(measured_lab - reference)  # Measure the distance between the bean colour and each reference Agtron sample in Lab space.

        if distance < smallest_distance:
            smallest_distance = distance  
            best_agtron = sample["agtron"]  

    return best_agtron, smallest_distance

def load_image(image_path: Path) -> np.ndarray:  # Read the roast image from disk and fail clearly if it is missing.
    """Load an image and raise a useful error if loading fails."""  
    image_bgr = cv2.imread(str(image_path))  # Load an image from disk using OpenCV.

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

    # Slight blur reduces isolated noisy pixels.
    blurred = cv2.GaussianBlur(image_bgr, (5, 5), 0)  # Smooth the image to reduce noise before processing.

    # HSV separates colour from brightness more effectively than grayscale.
    image_hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)  # Convert the image to a different colour space.
    image_lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

    # Approximate brown coffee-bean range.
    # lower_brown = np.array([2, 35, 35], dtype=np.uint8) # for hsv
    # upper_brown = np.array([30, 255, 245], dtype=np.uint8) # for hsv
    lower_brown = np.array([10*(255.0/100.0), -10 + 128.0, 5 + 128.0], dtype=np.uint8) #arbitrary values
    upper_brown = np.array([75*(255.0/100.0), 40 + 128.0, 60 + 128.0], dtype=np.uint8)

    # print(image_hsv)
    # print(image_lab)

    colour_mask = cv2.inRange(  # Mask only the brown coffee pixels that match the selected threshold range
        image_lab,          # change this to either hsv or lab
        lower_brown,  
        upper_brown,  
    )  

    # Limit processing to the central bean-containing region.
    roi_mask = np.zeros((height, width), dtype=np.uint8)  # Initialise a blank array that will be filled with the processed output.

    centre = (width // 2, (height // 2)-100)  

    # Adjust proportions to fit the circular bean region
    # rn these values (0.17 and 0,25 for w and h) work for file.good.jpg ----> might need small adjustments but works well overall
    axes = (  
        int(width * 0.17),  
        int(height * 0.25),  
    )  

    cv2.ellipse(  # Draw the fitted ellipse on the output image to visually verify the bean shape.
        roi_mask,  
        centre,  
        axes,  
        0,  
        0,  
        360,  
        255,  
        thickness=-1,  #fills the ellipse
    )  
    # image: Image on which ellipse is drawn.
    # center: Center of ellipse as (x, y).
    # axes: Tuple containing semi-major and semi-minor axis lengths (half of the full axis lengths).
    # angle: Rotation of ellipse in degrees.
    # startAngle: Starting angle of arc.
    # endAngle: Ending angle of arc.
    # color: BGR color tuple.
    # thickness: Border thickness (-1 fills the ellipse).

    # Keep only brown pixels inside the region of interest.
    mask = cv2.bitwise_and(colour_mask, roi_mask)  # Combine masks to isolate only the pixels that belong to the bean or overlap region.

    small_kernel = cv2.getStructuringElement(  # Create the element that defines how the mask is expanded or eroded during filtering.
        cv2.MORPH_ELLIPSE,  
        (3, 3),
    )  

    large_kernel = cv2.getStructuringElement(  # Create the element that defines how the mask is expanded or eroded during filtering.
        cv2.MORPH_ELLIPSE,  
        (5, 5),
    )  

    # Remove small isolated pixels.
    # mask = cv2.morphologyEx(  # Apply a morphological filter to clean or reshape the mask.
    #     mask,  
    #     cv2.MORPH_OPEN,  
    #     small_kernel,  
    #     iterations=1,  
    # )  

    # # Fill small holes and connect nearby bean regions.
    mask = cv2.morphologyEx(  # Apply a morphological filter to clean or reshape the mask.
        mask,  
        cv2.MORPH_CLOSE,  
        large_kernel,  
        iterations=2,  
    )  

    return mask


def measure_bean_colour(  # Read the Lab values of the masked bean pixels and summarise their colour as mean and median values.
    image_bgr: np.ndarray,  
    mask: np.ndarray,  
) -> tuple[np.ndarray, np.ndarray]:  
    """Return the median and mean CIELAB values of all segmented bean pixels."""  
    image_lab_opencv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)  # Convert the image to a different colour space.

    bean_pixels_opencv = image_lab_opencv[mask > 0]  

    if bean_pixels_opencv.size == 0:
        raise ValueError("The mask contains no bean pixels.")
    lab_pixels = bean_pixels_opencv.astype(np.float32)  
    
    bean_pixels_lab = np.empty_like(lab_pixels, dtype=np.float32)  
    bean_pixels_lab[:, 0] = lab_pixels[:, 0] * 100.0 / 255.0  
    bean_pixels_lab[:, 1] = lab_pixels[:, 1] - 128.0  
    bean_pixels_lab[:, 2] = lab_pixels[:, 2] - 128.0  

    median_lab = np.median(bean_pixels_lab, axis=0)  # Compute the median or mean Lab colour of the bean pixels so the roast colour can be summarised.
    mean_lab = np.mean(bean_pixels_lab, axis=0)  # Compute the median or mean Lab colour of the bean pixels so the roast colour can be summarised.

    return median_lab, mean_lab


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
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # Convert the image to a different colour space.

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
    f"Agtron = {estimated_agtron}\n"
    f"Distance = {error:.2f}"
    )  

    for axis in axes:  # Iterate over every item in the collection so each region is processed individually.
        axis.axis("off")  

    plt.tight_layout()  
    plt.show()  # Create or display the plotting figure.


def main() -> None:  # Run the full Otsu threshold experiment over the Lab channels.
    image_path = Path("images/file.good.jpg")  

    image_bgr = load_image(image_path)  
    mask = create_bean_mask(image_bgr)  

    median_lab, mean_lab = measure_bean_colour(  
        image_bgr,  
        mask,  
    )  

    print("Mean bean colour:")  # Print the current result to the console.
    print(f"L* = {mean_lab[0]:.2f}")  # Print the current result to the console.
    print(f"a* = {mean_lab[1]:.2f}")  # Print the current result to the console.
    print(f"b* = {mean_lab[2]:.2f}")  # Print the current result to the console.

    print("\nMedian bean colour:")  # Print the current result to the console.
    print(f"L* = {median_lab[0]:.2f}")  # Print the current result to the console.
    print(f"a* = {median_lab[1]:.2f}")  # Print the current result to the console.
    print(f"b* = {median_lab[2]:.2f}")  # Print the current result to the console.

    estimated_agtron, error = estimate_agtron(median_lab)  

    display_results(  
    image_bgr,  
    mask,  
    mean_lab,  
    median_lab,  
    estimated_agtron,  
    error  
    )  


if __name__ == "__main__":  # Run the script only when this file is executed directly.
    main()  

