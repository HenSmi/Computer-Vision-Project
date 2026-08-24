from pathlib import Path  # Let the script resolve file paths relative to the project folder.

import cv2  # Use OpenCV to read, filter, and display the roast images.
import numpy as np  # Work with image arrays and numeric operations during the analysis.


BASE_DIR = Path(__file__).resolve().parent  # Keep the script directory as the base for all file paths.  
IMAGE_PATH = BASE_DIR / "images" / "dark2.jpeg"  
RESULTS_DIR = BASE_DIR / "otsu_results"  
WINDOW = "CIELAB Otsu Threshold Experiment"  


def do_nothing(_: int) -> None:  # Required OpenCV callback so the trackbar controls remain interactive without extra logic.
    """Required callback function for OpenCV trackbars."""  
    pass  


def draw_histogram(channel: np.ndarray, threshold: float) -> np.ndarray:  # Draw the channel histogram and mark the Otsu threshold line on it.
    """Draw the channel histogram and mark the Otsu threshold."""  
    width = 768  
    height = 250  

    # Calculate the number of pixels at each intensity from 0 to 255.
    histogram = cv2.calcHist([channel], [0], None, [256], [0, 256]).flatten()

    # Scale the histogram so that its highest point fits in the output image.
    histogram /= max(histogram.max(), 1)  

    # Create a black image on which the histogram will be drawn.
    output = np.zeros((height, width, 3), dtype=np.uint8)  # Initialise a blank array that will be filled with the processed output.

    # Draw the histogram as connected white lines.
    for value in range(1, 256):  # Iterate over every item in the collection so each region is processed individually.
        x1 = int((value - 1) * width / 256)  
        x2 = int(value * width / 256)  
        y1 = height - int(histogram[value - 1] * (height - 20))  
        y2 = height - int(histogram[value] * (height - 20))  

        cv2.line(output, (x1, y1), (x2, y2), (255, 255, 255), 1)

    # Draw the automatically selected Otsu threshold as a red vertical line.
    threshold_x = int(threshold * width / 256)  
    cv2.line(output, (threshold_x, 0), (threshold_x, height - 1), (0, 0, 255), 2)

    # Add the threshold value to the histogram image.
    cv2.putText(output, f"Otsu threshold = {threshold:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return output


def add_information(
    image: np.ndarray,  
    channel_name: str,  
    threshold: float,  
    white_percentage: float,  
    kernel_size: int,  
    brightness: int,  
    contrast: float,  
    clahe_enabled: bool,  
) -> np.ndarray:  
    """Write the current processing settings onto the binary mask."""  
    output = image.copy()  

    lines = [  # Store the fixed reference values used throughout this analysis.
        f"Channel: {channel_name}",  
        f"Otsu threshold: {threshold:.1f}",  
        f"White pixels: {white_percentage:.2f}%",  
        f"Gaussian kernel: {kernel_size} x {kernel_size}",  
        f"Brightness: {brightness:+d}",  
        f"Contrast: {contrast:.2f}",  
        f"CLAHE: {'On' if clahe_enabled else 'Off'}",  
    ]  

    for index, text in enumerate(lines):  # Iterate over every item in the collection so each region is processed individually.
        cv2.putText(output, text, (15, 30 + index * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    return output


def fit_panel(image: np.ndarray, width: int = 450, height: int = 450) -> np.ndarray:  # Resize the image panels so the comparison view keeps a consistent layout.
    """Resize an image to the dimensions used in the display window."""  
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def save_current_results(  # Save the processed result, binary mask, and final display for the current threshold choice.
    channel_name: str,  
    processed: np.ndarray,  
    binary_mask: np.ndarray,  
    final_image: np.ndarray,  
) -> None:  
    """Save the latest processed, binary, and composite images to disk."""  
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)  

    safe_channel_name = channel_name.lower().replace(" ", "_").replace("*", "")  
    processed_path = RESULTS_DIR / f"otsu_processed_{safe_channel_name}.png"  
    binary_path = RESULTS_DIR / f"otsu_binary_{safe_channel_name}.png"  
    composite_path = RESULTS_DIR / f"otsu_complete_{safe_channel_name}.png"  

    written_processed = cv2.imwrite(str(processed_path), processed)  # Save the current image to a file.
    written_binary = cv2.imwrite(str(binary_path), binary_mask)  # Save the current image to a file.
    written_composite = cv2.imwrite(str(composite_path), final_image)  # Save the current image to a file.

    if written_processed and written_binary and written_composite:
        print(f"Results saved in: {RESULTS_DIR}")  # Print the current result to the console.
    else:  # Run this alternative branch when the previous condition fails.
        print("One or more outputs could not be saved.")  # Print the current result to the console.
        print(f"Processed saved: {written_processed} -> {processed_path}")  # Print the current result to the console.
        print(f"Binary saved: {written_binary} -> {binary_path}")  # Print the current result to the console.
        print(f"Composite saved: {written_composite} -> {composite_path}")  # Print the current result to the console.


def main() -> None:  # Run the full Otsu threshold experiment over the Lab channels.
    # Load the input image in OpenCV's BGR format.
    image_bgr = cv2.imread(str(IMAGE_PATH))  # Load an image from disk using OpenCV.

    if image_bgr is None:
        raise FileNotFoundError(f"Could not open image: {IMAGE_PATH}")

    # Convert the image from BGR to OpenCV's CIELAB representation.
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)  # Convert the image to a different colour space.

    # Separate the image into L*, a* and b* channels.
    channels = list(cv2.split(image_lab))  # Split the Lab image into separate channels so the Otsu threshold can be tuned per colour dimension.
    channel_names = ["L* lightness", "a* green-red", "b* blue-yellow"]  # Store the fixed reference values used throughout this analysis.

    # Create the display window.
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)  
    cv2.resizeWindow(WINDOW, 1500, 900)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)  

    # Display an initial image so the Windows GUI backend creates the window.
    initial_display = cv2.resize(image_bgr, (900, 700), interpolation=cv2.INTER_AREA)
    cv2.imshow(WINDOW, initial_display)  # Display the current result in the OpenCV window so the user can inspect it live.
    cv2.waitKey(100)  

    # Create the trackbars after the window has been displayed.
    cv2.createTrackbar("Channel 0=L 1=a 2=b", WINDOW, 0, 2, do_nothing)  # Add an interactive slider so the user can tune the current processing parameter live.
    cv2.createTrackbar("Gaussian radius", WINDOW, 2, 15, do_nothing)  # Add an interactive slider so the user can tune the current processing parameter live.
    cv2.createTrackbar("Brightness", WINDOW, 100, 200, do_nothing)  # Add an interactive slider so the user can tune the current processing parameter live.
    cv2.createTrackbar("Contrast x100", WINDOW, 100, 300, do_nothing)  # Add an interactive slider so the user can tune the current processing parameter live.
    cv2.createTrackbar("Invert", WINDOW, 1, 1, do_nothing)  # Add an interactive slider so the user can tune the current processing parameter live.
    cv2.createTrackbar("CLAHE", WINDOW, 0, 1, do_nothing)  # Add an interactive slider so the user can tune the current processing parameter live.
    cv2.createTrackbar("CLAHE clip x10", WINDOW, 20, 100, do_nothing)  # Add an interactive slider so the user can tune the current processing parameter live.
    cv2.createTrackbar("CLAHE grid", WINDOW, 8, 32, do_nothing)  # Add an interactive slider so the user can tune the current processing parameter live.

    cv2.resizeWindow(WINDOW, 1500, 900)
    cv2.waitKey(100)  

    while True:  # Keep the preview loop running until the user exits the window.
        # Read the current trackbar settings.
        channel_index = cv2.getTrackbarPos("Channel 0=L 1=a 2=b", WINDOW)  # Read the current slider value so the live preview matches the chosen settings.
        blur_radius = cv2.getTrackbarPos("Gaussian radius", WINDOW)  # Read the current slider value so the live preview matches the chosen settings.
        brightness_slider = cv2.getTrackbarPos("Brightness", WINDOW)  # Read the current slider value so the live preview matches the chosen settings.
        contrast_slider = cv2.getTrackbarPos("Contrast x100", WINDOW)  # Read the current slider value so the live preview matches the chosen settings.
        invert = cv2.getTrackbarPos("Invert", WINDOW)  # Read the current slider value so the live preview matches the chosen settings.
        clahe_enabled = cv2.getTrackbarPos("CLAHE", WINDOW)  # Read the current slider value so the live preview matches the chosen settings.
        clahe_clip_slider = cv2.getTrackbarPos("CLAHE clip x10", WINDOW)  # Read the current slider value so the live preview matches the chosen settings.
        clahe_grid_slider = cv2.getTrackbarPos("CLAHE grid", WINDOW)  # Read the current slider value so the live preview matches the chosen settings.

        # Select the requested CIELAB channel.
        processed = channels[channel_index].copy()  
        channel_name = channel_names[channel_index]  

        # Convert the blur radius to an odd Gaussian kernel size.
        kernel_size = 2 * blur_radius + 1  

        # Convert slider values to useful brightness and contrast values.
        brightness = brightness_slider - 100  
        contrast = max(contrast_slider, 1) / 100.0  

        # Smooth the selected channel before thresholding.
        processed = cv2.GaussianBlur(processed, (kernel_size, kernel_size), 0)  # Smooth the image to reduce noise before processing.

        # Apply the selected brightness offset while keeping values between 0 and 255.
        processed = np.clip(processed.astype(np.int16) + brightness, 0, 255).astype(np.uint8)  # Clamp the brightness-adjusted pixels back into the 0-255 range so the image remains valid.

        # Apply contrast scaling.
        processed = cv2.convertScaleAbs(processed, alpha=contrast, beta=0)  # Apply contrast scaling so the selected channel is easier to threshold accurately.

        # Apply local contrast enhancement when CLAHE is enabled.
        if clahe_enabled:
            clip_limit = max(clahe_clip_slider, 1) / 10.0  
            grid_size = max(clahe_grid_slider, 1)  
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))  # Create the CLAHE object that enhances local contrast without flattening the whole image.
            processed = clahe.apply(processed)  

        # Choose whether darker or brighter regions should become white.
        threshold_mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY  

        # Otsu automatically determines the threshold from the image histogram.
        otsu_threshold, binary_mask = cv2.threshold(processed, 0, 255, threshold_mode + cv2.THRESH_OTSU)  # Threshold the image to create a binary mask.

        # Calculate the percentage of the binary image that is white.
        white_percentage = np.count_nonzero(binary_mask) / binary_mask.size * 100.0  # Count the selected pixels for area or coverage calculations.

        # Convert grayscale images to BGR so they can be joined with the original.
        processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)  # Convert the image to a different colour space.
        binary_bgr = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)  # Convert the image to a different colour space.

        # Add the current settings to the binary image.
        binary_bgr = add_information(  # Overlay the current operation settings on the processed image for easier interpretation.
            binary_bgr,  
            channel_name,  
            otsu_threshold,  
            white_percentage,  
            kernel_size,  
            brightness,  
            contrast,  
            bool(clahe_enabled),  
        )  

        # Resize all three panels to the same dimensions.
        original_panel = fit_panel(image_bgr)  
        processed_panel = fit_panel(processed_bgr)  
        binary_panel = fit_panel(binary_bgr)  

        # Place the original, processed and thresholded images side by side.
        top_row = np.hstack([original_panel, processed_panel, binary_panel])  # Combine image panels into a larger display array.

        # Draw and resize the histogram to match the width of the top row.
        histogram = draw_histogram(processed, otsu_threshold)  
        histogram = cv2.resize(histogram, (top_row.shape[1], 250))

        # Combine the image panels and histogram into one final image.
        final_image = np.vstack([top_row, histogram])  # Combine image panels into a larger display array.

        # Display the final image.
        cv2.imshow(WINDOW, final_image)  # Display the current result in the OpenCV window so the user can inspect it live.

        key = cv2.waitKey(20) & 0xFF  

        # Press Q or Escape to close the program.
        if key in (ord("q"), 27):
            save_current_results(channel_name, processed, binary_mask, final_image)  
            break  

        # Press S to save the current results.
        if key == ord("s"):
            save_current_results(channel_name, processed, binary_mask, final_image)  

    cv2.destroyAllWindows()  


if __name__ == "__main__":  # Run the script only when this file is executed directly.
    main()  

