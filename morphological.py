from pathlib import Path  # Let the script resolve file paths relative to the project folder.

import cv2  # Use OpenCV to read, filter, and display the roast images.
import numpy as np  # Work with image arrays and numeric operations during the analysis.


BASE_DIR = Path(__file__).resolve().parent  # Keep the script directory as the base for all file paths.  
IMAGE_PATH = BASE_DIR / "otsu_results" / "otsu_medium.png"  
RESULTS_DIR = BASE_DIR / "morphology_results"  
WINDOW = "Morphological Operations Experiment"  


def do_nothing(_: int) -> None:  # Required OpenCV callback so the trackbar controls remain interactive without extra logic.
    """Required callback for OpenCV trackbars."""  
    pass  


def add_information(image: np.ndarray, operation: str, shape: str, kernel_size: int, iterations: int, white_percentage: float) -> np.ndarray:
    """Write the current morphology settings onto the output image."""  
    output = image.copy()  

    lines = [  # Store the fixed reference values used throughout this analysis.
        f"Operation: {operation}",  
        f"Kernel shape: {shape}",  
        f"Kernel size: {kernel_size} x {kernel_size}",  
        f"Iterations: {iterations}",  
        f"White pixels: {white_percentage:.2f}%",  
        "Press S to save",  
        "Press Q or Esc to exit",  
    ]  

    for index, text in enumerate(lines):  # Iterate over every item in the collection so each region is processed individually.
        cv2.putText(output, text, (15, 30 + index * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    return output


def apply_morphology(binary: np.ndarray, operation_index: int, kernel: np.ndarray, iterations: int) -> np.ndarray:
    """Apply the selected morphological operation."""  
    if operation_index == 0:
        return binary.copy()

    if operation_index == 1:
        return cv2.erode(binary, kernel, iterations=iterations)

    if operation_index == 2:
        return cv2.dilate(binary, kernel, iterations=iterations)

    if operation_index == 3:
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)  # Clean the mask with a morphological operation to remove noise or close gaps.

    if operation_index == 4:
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)  # Clean the mask with a morphological operation to remove noise or close gaps.

    if operation_index == 5:
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)  # Apply a morphological filter to clean or reshape the mask.
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)  # Clean the mask with a morphological operation to remove noise or close gaps.

    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)  # Apply a morphological filter to clean or reshape the mask.
    return cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=iterations)  # Clean the mask with a morphological operation to remove noise or close gaps.


def main() -> None:  # Run the full Otsu threshold experiment over the Lab channels.
    # Load the saved Otsu mask as a single-channel grayscale image.
    original = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)  # Load an image from disk using OpenCV.

    if original is None:
        raise FileNotFoundError(f"Could not open image: {IMAGE_PATH}")

    # Force the loaded image to contain only 0 and 255.
    _, binary = cv2.threshold(original, 127, 255, cv2.THRESH_BINARY)  # Threshold the image to create a binary mask.

    operation_names = [  # Store the fixed reference values used throughout this analysis.
        "None",  
        "Erosion",  
        "Dilation",  
        "Opening",  
        "Closing",  
        "Opening then closing",  
        "Closing then opening",  
    ]  

    shape_names = ["Rectangle", "Ellipse", "Cross"]  # Store the fixed reference values used throughout this analysis.
    shape_types = [cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, cv2.MORPH_CROSS]  # Store the fixed reference values used throughout this analysis.

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)  
    cv2.resizeWindow(WINDOW, 1400, 750)

    # Operation values:
    # 0=None, 1=Erosion, 2=Dilation, 3=Opening, 4=Closing,
    # 5=Opening then closing, 6=Closing then opening.
    cv2.createTrackbar("Operation 0-6", WINDOW, 3, 6, do_nothing)  # Add an interactive slider so the user can tune the current processing parameter live.

    # Kernel shape values: 0=Rectangle, 1=Ellipse, 2=Cross.
    cv2.createTrackbar("Kernel shape 0-2", WINDOW, 1, 2, do_nothing)  # Add an interactive slider so the user can tune the current processing parameter live.

    # Radius is converted to an odd kernel size: 0->1, 1->3, 2->5, etc.
    cv2.createTrackbar("Kernel radius", WINDOW, 1, 15, do_nothing)  # Add an interactive slider so the user can tune the current processing parameter live.

    cv2.createTrackbar("Iterations", WINDOW, 1, 10, do_nothing)  # Add an interactive slider so the user can tune the current processing parameter live.

    # Invert allows you to correct a mask where beans are black instead of white.
    cv2.createTrackbar("Invert input", WINDOW, 0, 1, do_nothing)  # Add an interactive slider so the user can tune the current processing parameter live.

    while True:  # Keep the preview loop running until the user exits the window.
        operation_index = cv2.getTrackbarPos("Operation 0-6", WINDOW)  # Read the current slider value so the live preview matches the chosen settings.
        shape_index = cv2.getTrackbarPos("Kernel shape 0-2", WINDOW)  # Read the current slider value so the live preview matches the chosen settings.
        kernel_radius = cv2.getTrackbarPos("Kernel radius", WINDOW)  # Read the current slider value so the live preview matches the chosen settings.
        iterations = max(cv2.getTrackbarPos("Iterations", WINDOW), 1)  # Read the current slider value so the live preview matches the chosen settings.
        invert = cv2.getTrackbarPos("Invert input", WINDOW)  # Read the current slider value so the live preview matches the chosen settings.

        current_input = cv2.bitwise_not(binary) if invert else binary.copy()  # Invert the mask when the user wants dark beans treated as the foreground.

        # Morphological kernels normally use odd dimensions.
        kernel_size = 2 * kernel_radius + 1  
        kernel = cv2.getStructuringElement(shape_types[shape_index], (kernel_size, kernel_size))  # Create the element that defines how the mask is expanded or eroded during filtering.

        processed = apply_morphology(current_input, operation_index, kernel, iterations)  # Run the selected erosion, dilation, or opening/closing sequence on the current bean mask.

        white_percentage = np.count_nonzero(processed) / processed.size * 100.0  # Count the selected pixels for area or coverage calculations.

        input_bgr = cv2.cvtColor(current_input, cv2.COLOR_GRAY2BGR)  # Convert the image to a different colour space.
        processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)  # Convert the image to a different colour space.

        processed_bgr = add_information(  # Overlay the current operation settings on the processed image for easier interpretation.
            processed_bgr,  
            operation_names[operation_index],  
            shape_names[shape_index],  
            kernel_size,  
            iterations,  
            white_percentage,  
        )  

        panel_width = 600  
        panel_height = 600  

        input_panel = cv2.resize(input_bgr, (panel_width, panel_height), interpolation=cv2.INTER_NEAREST)
        output_panel = cv2.resize(processed_bgr, (panel_width, panel_height), interpolation=cv2.INTER_NEAREST)

        final_display = np.hstack([input_panel, output_panel])  # Combine image panels into a larger display array.

        cv2.imshow(WINDOW, final_display)  # Display the current result in the OpenCV window so the user can inspect it live.

        key = cv2.waitKey(20) & 0xFF  

        if key in (ord("q"), 27):
            break  

        if key == ord("s"):
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)  

            cv2.imwrite(str(RESULTS_DIR / "morphology_output.png"), processed)  # Save the current image to a file.
            cv2.imwrite(str(RESULTS_DIR / "morphology_comparison.png"), final_display)  # Save the current image to a file.
            cv2.imwrite(str(RESULTS_DIR / "morphology_kernel.png"), kernel * 255)  # Save the current image to a file.

            print("\nMorphology result saved")  # Print the current result to the console.
            print(f"Operation:   {operation_names[operation_index]}")  # Print the current result to the console.
            print(f"Shape:       {shape_names[shape_index]}")  # Print the current result to the console.
            print(f"Kernel:      {kernel_size} x {kernel_size}")  # Print the current result to the console.
            print(f"Iterations:  {iterations}")  # Print the current result to the console.
            print(f"Output path: {RESULTS_DIR}")  # Print the current result to the console.

    cv2.destroyAllWindows()  


if __name__ == "__main__":  # Run the script only when this file is executed directly.
    main()  

