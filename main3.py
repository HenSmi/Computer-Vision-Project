from pathlib import Path  # Let the script resolve file paths relative to the project folder.

import cv2  # Use OpenCV to read, filter, and display the roast images.
import matplotlib.pyplot as plt  # Plot the processed image and measurement panels for inspection.
import numpy as np  # Work with image arrays and numeric operations during the analysis.

from main import create_bean_mask, load_image


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIRECTORY = Path(__file__).resolve().parent  
IMAGE_PATH = SCRIPT_DIRECTORY / "images" / "medium.jpg"  

# Optional calibration:
# Replace this with your measured number of millimetres per pixel.
# Leave as None until the camera has been calibrated.
MM_PER_PIXEL: float | None = None  


# ---------------------------------------------------------------------------
# Region of interest
# ---------------------------------------------------------------------------

def create_viewing_region_mask(
    image_shape: tuple[int, ...],  
) -> np.ndarray:  
    """  
    Create a fixed elliptical mask representing the useful roaster window.  

    Only pixels inside this mask are included when calculating area fraction.  
    Adjust the centre and axes to match the actual viewing region.  
    """  
    height, width = image_shape[:2]  

    roi_mask = np.zeros((height, width), dtype=np.uint8)  # Initialise a blank array that will be filled with the processed output.

    centre = (  
        width // 2,  
        height // 2,  
    )  

    axes = (  
        int(width * 0.40),  
        int(height * 0.45),  
    )  

    cv2.ellipse(  # Draw the fitted ellipse on the output image to visually verify the bean shape.
        roi_mask,  
        centre,  
        axes,  
        angle=0,  
        startAngle=0,  
        endAngle=360,  
        color=255,  
        thickness=-1,  
    )  

    return roi_mask


# ---------------------------------------------------------------------------
# Area calculation
# ---------------------------------------------------------------------------

def calculate_projected_area(
    bean_mask: np.ndarray,  
    roi_mask: np.ndarray,  
    mm_per_pixel: float | None = None,  
) -> dict[str, float | int | None]:  
    """  
    Calculate the total visible projected bean area.  

    The bean mask and region-of-interest mask must contain:  
        255 for selected pixels  
        0 for rejected pixels  
    """  
    if bean_mask.shape != roi_mask.shape:
        raise ValueError(
            "The bean mask and ROI mask must have the same dimensions."  
        )  

    # Ensure that only detected bean pixels inside the viewing region count.
    valid_bean_mask = cv2.bitwise_and(  # Combine masks to isolate only the pixels that belong to the bean or overlap region.
        bean_mask,  
        roi_mask,  
    )  

    bean_pixel_area = int(  
        np.count_nonzero(valid_bean_mask)  # Count the selected pixels for area or coverage calculations.
    )  

    roi_pixel_area = int(  
        np.count_nonzero(roi_mask)  # Count the selected pixels for area or coverage calculations.
    )  

    if roi_pixel_area == 0:
        raise ValueError("The viewing-region mask contains no pixels.")

    area_fraction = bean_pixel_area / roi_pixel_area  
    coverage_percentage = area_fraction * 100.0  

    projected_area_mm2 = None  

    if mm_per_pixel is not None:
        if mm_per_pixel <= 0:
            raise ValueError("MM_PER_PIXEL must be greater than zero.")

        # One image pixel represents a square with this physical area.
        mm2_per_pixel = mm_per_pixel**2  

        projected_area_mm2 = bean_pixel_area * mm2_per_pixel  

    return {
        "bean_pixels": bean_pixel_area,  
        "roi_pixels": roi_pixel_area,  
        "area_fraction": area_fraction,  
        "coverage_percentage": coverage_percentage,  
        "projected_area_mm2": projected_area_mm2,  
        "valid_bean_mask": valid_bean_mask,  
    }  


def calculate_relative_expansion(
    current_area: float,  
    initial_area: float,  
) -> float:  
    """  
    Calculate projected-area change relative to the initial roast image.  
    """  
    if initial_area <= 0:
        raise ValueError("Initial area must be greater than zero.")

    return (
        (current_area - initial_area)
        / initial_area  
        * 100.0  
    )  


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def create_area_overlay(
    image_bgr: np.ndarray,  
    valid_bean_mask: np.ndarray,  
    roi_mask: np.ndarray,  
) -> np.ndarray:  
    """  
    Overlay the selected projected bean area and ROI boundary on the image.  
    """  
    image_rgb = cv2.cvtColor(  # Convert the image to a different colour space.
        image_bgr,  
        cv2.COLOR_BGR2RGB,  
    )  

    overlay = image_rgb.copy()  

    # Create a visible overlay for selected bean pixels.
    highlighted = image_rgb.copy()  
    highlighted[valid_bean_mask > 0] = np.array(
        [255, 255, 255],  
        dtype=np.uint8,  
    )  

    overlay = cv2.addWeighted(  # Blend the original image with the highlighted bean mask to create the overlay view.
        image_rgb,  
        0.65,  
        highlighted,  
        0.35,  
        0,  
    )  

    # Draw the boundary of the valid viewing region.
    roi_contours, _ = cv2.findContours(  # Find the outlines of each bean-shaped region so they can be measured individually.
        roi_mask,  
        cv2.RETR_EXTERNAL,  
        cv2.CHAIN_APPROX_SIMPLE,  
    )  

    cv2.drawContours(  # Draw the bean boundary onto the output image to highlight the detected component.
        overlay,  
        roi_contours,  
        contourIdx=-1,  
        color=(255, 255, 255),  
        thickness=3,  
    )  

    return overlay


def display_area_results(  # Display the original image, ROI mask, bean area mask, and area summary together.
    image_bgr: np.ndarray,  
    bean_mask: np.ndarray,  
    roi_mask: np.ndarray,  
    measurements: dict[str, float | int | None],  
) -> None:  
    """  
    Display the original image, masks, area overlay and measurements.  
    """  
    image_rgb = cv2.cvtColor(  # Convert the image to a different colour space.
        image_bgr,  
        cv2.COLOR_BGR2RGB,  
    )  

    valid_bean_mask = measurements["valid_bean_mask"]  

    overlay = create_area_overlay(  # Create the overlay image that shows the bean coverage inside the fixed roaster window.
        image_bgr,  
        valid_bean_mask,  
        roi_mask,  
    )  

    figure, axes = plt.subplots(  # Create or display the plotting figure.
        1,  
        4,  
        figsize=(18, 5),  
    )  

    axes[0].imshow(image_rgb)  
    axes[0].set_title("Original image")  
    axes[0].axis("off")  

    axes[1].imshow(roi_mask, cmap="gray")  
    axes[1].set_title("Fixed viewing region")  
    axes[1].axis("off")  

    axes[2].imshow(valid_bean_mask, cmap="gray")  
    axes[2].set_title("Visible bean-area mask")  
    axes[2].axis("off")  

    axes[3].imshow(overlay)  

    measurement_text = (  
        f"Bean area: {measurements['bean_pixels']:,} pixels²\n"  
        f"ROI area: {measurements['roi_pixels']:,} pixels²\n"  
        f"Coverage: {measurements['coverage_percentage']:.2f}%"  
    )  

    if measurements["projected_area_mm2"] is not None:
        measurement_text += (  
            "\n"  
            f"Projected area: "  
            f"{measurements['projected_area_mm2']:.2f} mm²"  
        )  

    axes[3].set_title(  
        "Measured projected area\n"  
        + measurement_text  
    )  
    axes[3].axis("off")  

    plt.tight_layout()  
    plt.show()  # Create or display the plotting figure.


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def main() -> None:  # Run the full Otsu threshold experiment over the Lab channels.
    image_bgr = load_image(IMAGE_PATH)  # Load the source roast image so the bean mask and area analysis can be run.

    # Reuse the segmentation developed for colour analysis.
    bean_mask = create_bean_mask(image_bgr)  

    # Define the fixed part of the roaster window where beans can appear.
    roi_mask = create_viewing_region_mask(  
        image_bgr.shape  
    )  

    measurements = calculate_projected_area(  
        bean_mask=bean_mask,  
        roi_mask=roi_mask,  
        mm_per_pixel=MM_PER_PIXEL,  
    )  

    print("\nProjected-area measurement")  # Print the current result to the console.
    print("--------------------------")  # Print the current result to the console.
    print(  # Print the current result to the console.
        f"Bean area:       "  
        f"{measurements['bean_pixels']:,} pixels²"  
    )  
    print(  # Print the current result to the console.
        f"Viewing area:    "  
        f"{measurements['roi_pixels']:,} pixels²"  
    )  
    print(  # Print the current result to the console.
        f"Area fraction:   "  
        f"{measurements['area_fraction']:.6f}"  
    )  
    print(  # Print the current result to the console.
        f"Coverage:        "  
        f"{measurements['coverage_percentage']:.2f}%"  
    )  

    if measurements["projected_area_mm2"] is not None:
        print(  # Print the current result to the console.
            f"Physical area:   "  
            f"{measurements['projected_area_mm2']:.2f} mm²"  
        )  
    else:  # Run this alternative branch when the previous condition fails.
        print(  # Print the current result to the console.
            "Physical area:   Not calculated "  
            "(camera calibration required)"  
        )  

    display_area_results(  
        image_bgr=image_bgr,  
        bean_mask=bean_mask,  
        roi_mask=roi_mask,  
        measurements=measurements,  
    )  


if __name__ == "__main__":  # Run the script only when this file is executed directly.
    main()  

