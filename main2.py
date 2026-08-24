from pathlib import Path  # Let the script resolve file paths relative to the project folder.

import cv2  # Use OpenCV to read, filter, and display the roast images.
import matplotlib.pyplot as plt  # Plot the processed image and measurement panels for inspection.
import numpy as np  # Work with image arrays and numeric operations during the analysis.
from skimage.feature import (  # Pull in the texture-analysis tools for LBP and GLCM.
    graycomatrix,  # Compute the grey-level co-occurrence matrix for each texture patch to capture bean texture.
    graycoprops,  # Extract the GLCM statistics that summarise contrast, uniformity, and texture strength.
    local_binary_pattern,  # Compute the local texture pattern for each bean pixel so the script can analyse surface roughness.
)  

# Reuse the loading and segmentation functions from main.py.
from main import create_bean_mask, load_image


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIRECTORY = Path(__file__).resolve().parent  
IMAGE_PATH = SCRIPT_DIRECTORY / "images" / "medium.jpg"  

# LBP parameters.
LBP_RADIUS = 2  
LBP_POINTS = 8 * LBP_RADIUS  
LBP_METHOD = "uniform"  

# GLCM parameters.
GLCM_LEVELS = 32  
GLCM_DISTANCES = [1, 2, 3]
GLCM_ANGLES = [  # Store the fixed reference values used throughout this analysis.
    0,  
    np.pi / 4,  
    np.pi / 2,  
    3 * np.pi / 4,  
]  

# GLCM is calculated from image patches that contain mostly bean pixels.
PATCH_SIZE = 64  
MINIMUM_BEAN_COVERAGE = 0.85  


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def prepare_grayscale(
    image_bgr: np.ndarray,  
) -> np.ndarray:  
    """  
    Convert the source image to an 8-bit grayscale image.  

    A small Gaussian blur suppresses isolated camera noise while preserving  
    larger surface features such as cracks and rough bean structure.  
    """  
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)  # Convert the image to a different colour space.

    image_gray = cv2.GaussianBlur(  # Smooth the image to reduce noise before processing.
        image_gray,  
        (3, 3),
        0,  
    )  

    return image_gray


def create_segmented_grayscale(
    image_gray: np.ndarray,  
    mask: np.ndarray,  
) -> np.ndarray:  
    """  
    Produce a grayscale image where non-bean pixels are displayed as black.  
    """  
    segmented_gray = np.zeros_like(image_gray)
    segmented_gray[mask > 0] = image_gray[mask > 0]  

    return segmented_gray


# ---------------------------------------------------------------------------
# Local Binary Pattern analysis
# ---------------------------------------------------------------------------

def calculate_lbp(
    image_gray: np.ndarray,  
    mask: np.ndarray,  
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  
    """  
    Calculate a uniform LBP representation and its normalised histogram.  

    Only LBP values whose centre pixels lie inside the bean mask are included  
    in the histogram.  
    """  
    lbp_image = local_binary_pattern(  # Compute the local texture pattern for each bean pixel so the script can analyse surface roughness.
        image_gray,  
        P=LBP_POINTS,  
        R=LBP_RADIUS,  
        method=LBP_METHOD,  
    )  

    valid_lbp_values = lbp_image[mask > 0]  

    if valid_lbp_values.size == 0:
        raise ValueError("No bean pixels are available for LBP analysis.")

    # Uniform LBP with P neighbours has P + 2 output categories.
    number_of_bins = LBP_POINTS + 2  

    histogram, bin_edges = np.histogram(  # Count the LBP values into bins so the texture histogram can be compared numerically.
        valid_lbp_values,  
        bins=np.arange(0, number_of_bins + 1),  
        range=(0, number_of_bins),  
    )  

    histogram = histogram.astype(np.float64)  

    if histogram.sum() > 0:
        histogram /= histogram.sum()  

    # Create a display image where only masked LBP pixels are visible.
    lbp_display = np.zeros_like(lbp_image, dtype=np.float32)
    lbp_display[mask > 0] = lbp_image[mask > 0]  

    return lbp_display, histogram, bin_edges


# ---------------------------------------------------------------------------
# GLCM analysis
# ---------------------------------------------------------------------------

def quantise_grayscale(
    image_gray: np.ndarray,  
    levels: int,  
) -> np.ndarray:  
    """  
    Quantise an 8-bit grayscale image to the requested number of GLCM levels.  

    For 32 levels:  
        0...255 becomes 0...31.  
    """  
    if levels < 2 or levels > 256:
        raise ValueError("GLCM levels must be between 2 and 256.")

    quantised = (  
        image_gray.astype(np.uint16) * levels // 256  
    )  

    quantised = np.clip(  # Clamp the brightness-adjusted pixels back into the 0-255 range so the image remains valid.
        quantised,  
        0,  
        levels - 1,  
    )  

    return quantised.astype(np.uint8)


def extract_valid_texture_patches(
    image_gray: np.ndarray,  
    mask: np.ndarray,  
    patch_size: int,  
    minimum_coverage: float,  
) -> list[np.ndarray]:  
    """  
    Extract patches containing mostly bean pixels.  

    Calculating a GLCM directly on a black-background segmented image would  
    cause the artificial black background to dominate the GLCM. Patches with  
    high bean-mask coverage reduce that problem.  
    """  
    height, width = image_gray.shape  

    valid_patches: list[np.ndarray] = []

    for top in range(0, height - patch_size + 1, patch_size):  # Iterate over every item in the collection so each region is processed individually.
        for left in range(0, width - patch_size + 1, patch_size):  # Iterate over every item in the collection so each region is processed individually.
            bottom = top + patch_size  
            right = left + patch_size  

            mask_patch = mask[top:bottom, left:right]  

            bean_coverage = np.count_nonzero(mask_patch) / mask_patch.size  # Count the selected pixels for area or coverage calculations.

            if bean_coverage >= minimum_coverage:
                image_patch = image_gray[top:bottom, left:right]  
                valid_patches.append(image_patch)  

    return valid_patches


def calculate_glcm_features(
    image_gray: np.ndarray,  
    mask: np.ndarray,  
) -> tuple[dict[str, float], int]:  
    """  
    Calculate averaged GLCM properties from valid bean-texture patches.  

    A GLCM is calculated for each accepted patch using several pixel distances  
    and four directions. Feature values are then averaged across all patches,  
    distances and directions.  
    """  
    patches = extract_valid_texture_patches(  
        image_gray=image_gray,  
        mask=mask,  
        patch_size=PATCH_SIZE,  
        minimum_coverage=MINIMUM_BEAN_COVERAGE,  
    )  

    if not patches:
        raise ValueError(
            "No valid GLCM patches were found. "  
            "Reduce MINIMUM_BEAN_COVERAGE or PATCH_SIZE."  
        )  

    feature_names = [
        "contrast",  
        "dissimilarity",  
        "homogeneity",  
        "energy",  
        "correlation",  
        "ASM",  
    ]  

    patch_results: dict[str, list[float]] = {
        feature_name: []  
        for feature_name in feature_names  # Iterate over every item in the collection so each region is processed individually.
    }  

    for patch in patches:  # Iterate over every item in the collection so each region is processed individually.
        quantised_patch = quantise_grayscale(  
            patch,  
            levels=GLCM_LEVELS,  
        )  

        glcm = graycomatrix(  # Compute the grey-level co-occurrence matrix for each texture patch to capture bean texture.
            quantised_patch,  
            distances=GLCM_DISTANCES,  
            angles=GLCM_ANGLES,  
            levels=GLCM_LEVELS,  
            symmetric=True,  
            normed=True,  
        )  

        for feature_name in feature_names:  # Iterate over every item in the collection so each region is processed individually.
            property_values = graycoprops(  # Extract the GLCM statistics that summarise contrast, uniformity, and texture strength.
                glcm,  
                feature_name,  
            )  

            patch_results[feature_name].append(  
                float(np.mean(property_values))  
            )  

    averaged_features = {
        feature_name: float(np.mean(values))  
        for feature_name, values in patch_results.items()  # Iterate over every item in the collection so each region is processed individually.
    }  

    return averaged_features, len(patches)


def create_patch_overlay(
    image_rgb: np.ndarray,  
    mask: np.ndarray,  
) -> tuple[np.ndarray, int]:  
    """  
    Draw rectangles around the image patches used for GLCM analysis.  
    """  
    overlay = image_rgb.copy()  

    height, width = mask.shape  
    patch_count = 0  

    for top in range(0, height - PATCH_SIZE + 1, PATCH_SIZE):  # Iterate over every item in the collection so each region is processed individually.
        for left in range(0, width - PATCH_SIZE + 1, PATCH_SIZE):  # Iterate over every item in the collection so each region is processed individually.
            bottom = top + PATCH_SIZE  
            right = left + PATCH_SIZE  

            mask_patch = mask[top:bottom, left:right]  
            coverage = np.count_nonzero(mask_patch) / mask_patch.size  # Count the selected pixels for area or coverage calculations.

            if coverage >= MINIMUM_BEAN_COVERAGE:
                cv2.rectangle(  
                    overlay,  
                    (left, top),
                    (right - 1, bottom - 1),
                    (255, 255, 255),
                    thickness=2,  
                )  

                patch_count += 1  

    return overlay, patch_count


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_texture_results(
    image_bgr: np.ndarray,  
    mask: np.ndarray,  
    image_gray: np.ndarray,  
    lbp_display: np.ndarray,  
    lbp_histogram: np.ndarray,  
    glcm_features: dict[str, float],  
    number_of_patches: int,  
) -> None:  
    """  
    Display the segmentation, LBP output, LBP histogram and GLCM features.  
    """  
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # Convert the image to a different colour space.

    segmented_gray = create_segmented_grayscale(  
        image_gray,  
        mask,  
    )  

    patch_overlay, _ = create_patch_overlay(  
        image_rgb,  
        mask,  
    )  

    figure, axes = plt.subplots(  # Create or display the plotting figure.
        2,  
        3,  
        figsize=(16, 10),  
    )  

    # Original image.
    axes[0, 0].imshow(image_rgb)  
    axes[0, 0].set_title("Original image")  
    axes[0, 0].axis("off")  

    # Bean mask.
    axes[0, 1].imshow(mask, cmap="gray")  
    axes[0, 1].set_title("Bean mask")  
    axes[0, 1].axis("off")  

    # Segmented grayscale image.
    axes[0, 2].imshow(segmented_gray, cmap="gray")  
    axes[0, 2].set_title("Segmented grayscale")  
    axes[0, 2].axis("off")  

    # LBP image.
    axes[1, 0].imshow(  
        lbp_display,  
        cmap="gray",  
        vmin=0,  
        vmax=LBP_POINTS + 1,  
    )  
    axes[1, 0].set_title(  
        f"Uniform LBP\n"  
        f"P = {LBP_POINTS}, R = {LBP_RADIUS}"
    )  
    axes[1, 0].axis("off")  

    # LBP histogram.
    pattern_numbers = np.arange(len(lbp_histogram))  # Build the x-axis for the LBP histogram so each bin is mapped to a texture pattern number.

    axes[1, 1].bar(  
        pattern_numbers,  
        lbp_histogram,  
    )  

    axes[1, 1].set_title("Normalised LBP histogram")  
    axes[1, 1].set_xlabel("Uniform LBP pattern")  
    axes[1, 1].set_ylabel("Normalised frequency")  
    axes[1, 1].set_xlim(  
        -0.5,  
        len(lbp_histogram) - 0.5,  
    )  
    axes[1, 1].grid(  
        axis="y",  
        alpha=0.3,  
    )  

    # GLCM patch overlay and feature values.
    axes[1, 2].imshow(patch_overlay)  
    axes[1, 2].set_title(  
        f"GLCM texture patches\n"  
        f"{number_of_patches} patches used"  
    )  
    axes[1, 2].axis("off")  

    glcm_text = (  
        f"Contrast:       {glcm_features['contrast']:.4f}\n"  
        f"Dissimilarity:  {glcm_features['dissimilarity']:.4f}\n"  
        f"Homogeneity:    {glcm_features['homogeneity']:.4f}\n"  
        f"Energy:         {glcm_features['energy']:.4f}\n"  
        f"Correlation:    {glcm_features['correlation']:.4f}\n"  
        f"ASM:            {glcm_features['ASM']:.4f}"  
    )  

    axes[1, 2].text(  
        0.02,  
        0.02,  
        glcm_text,  
        transform=axes[1, 2].transAxes,  
        verticalalignment="bottom",  
        horizontalalignment="left",  
        fontsize=10,  
        family="monospace",  
        bbox={  
            "facecolor": "white",  
            "alpha": 0.85,  
            "edgecolor": "black",  
        },  
    )  

    plt.tight_layout()  
    plt.show()  # Create or display the plotting figure.


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def main() -> None:  # Run the full Otsu threshold experiment over the Lab channels.
    image_bgr = load_image(IMAGE_PATH)  # Load the source roast image so the bean mask and area analysis can be run.

    # This uses the same segmentation method as the colour program.
    mask = create_bean_mask(image_bgr)  

    image_gray = prepare_grayscale(image_bgr)  # Convert the image to a single-channel grayscale form for texture analysis.

    lbp_display, lbp_histogram, _ = calculate_lbp(  # Compute the LBP texture features for bean pixels so the surface roughness can be measured.
        image_gray,  
        mask,  
    )  

    glcm_features, number_of_patches = calculate_glcm_features(  # Compute the GLCM texture features across valid patches so the bean texture can be summarised.
        image_gray,  
        mask,  
    )  

    print("\nLBP texture analysis")  # Print the current result to the console.
    print("--------------------")  # Print the current result to the console.
    print(f"Radius:             {LBP_RADIUS}")  # Print the current result to the console.
    print(f"Neighbour points:   {LBP_POINTS}")  # Print the current result to the console.
    print(f"Histogram bins:     {len(lbp_histogram)}")  # Print the current result to the console.

    print("\nGLCM texture analysis")  # Print the current result to the console.
    print("---------------------")  # Print the current result to the console.
    print(f"Grey levels:        {GLCM_LEVELS}")  # Print the current result to the console.
    print(f"Distances:          {GLCM_DISTANCES}")  # Print the current result to the console.
    print(f"Directions:         {len(GLCM_ANGLES)}")  # Print the current result to the console.
    print(f"Valid patches:      {number_of_patches}")  # Print the current result to the console.

    print("\nGLCM properties")  # Print the current result to the console.
    print("----------------")  # Print the current result to the console.

    for feature_name, feature_value in glcm_features.items():  # Iterate over every item in the collection so each region is processed individually.
        print(f"{feature_name:<16}: {feature_value:.6f}")  # Print the current result to the console.

    display_texture_results(  
        image_bgr=image_bgr,  
        mask=mask,  
        image_gray=image_gray,  
        lbp_display=lbp_display,  
        lbp_histogram=lbp_histogram,  
        glcm_features=glcm_features,  
        number_of_patches=number_of_patches,  
    )  


if __name__ == "__main__":  # Run the script only when this file is executed directly.
    main()  

