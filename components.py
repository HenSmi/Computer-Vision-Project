from pathlib import Path  # Let the script resolve file paths relative to the project folder.

import cv2  # Use OpenCV to read, filter, and display the roast images.
import numpy as np  # Work with image arrays and numeric operations during the analysis.


BASE_DIR = Path(__file__).resolve().parent  # Keep the script directory as the base for all file paths.  
IMAGE_PATH = BASE_DIR / "morphology_results" / "morphology_output.png"  # Point to the binary mask produced by the morphology experiment.  
RESULTS_DIR = BASE_DIR / "component_results"  # Set the output folder for the bean-component analysis results.  
WINDOW = "Potential Bean Components"  # Name the preview window used to inspect connected bean candidates.  


def do_nothing(_: int) -> None:  # Required OpenCV callback so the trackbar controls remain interactive without extra logic.
    pass  


def calculate_circularity(area: float, perimeter: float) -> float:  # Measure how close the bean contour is to a perfect circle.
    """Return 1.0 for a perfect circle and lower values for irregular shapes."""  
    if perimeter == 0:
        return 0.0

    return 4.0 * np.pi * area / perimeter**2


def calculate_ellipse_match(component_mask: np.ndarray, contour: np.ndarray):  # Fit an ellipse to the contour and compare it to the bean component.
    """Fit an ellipse and compare its shape with the connected component."""  
    if len(contour) < 5:
        return 0.0, 0.0, 0.0, None

    ellipse = cv2.fitEllipse(contour)  # Fit an ellipse to the contour so the script can compare the bean shape to a natural bean profile.

    # Fill the external contour so internal holes do not reduce the shape score.
    filled_component = np.zeros_like(component_mask)
    cv2.drawContours(filled_component, [contour], -1, 255, -1)  # Draw the bean boundary onto the output image to highlight the detected component.

    # Create a filled mask of the fitted ellipse.
    ellipse_mask = np.zeros_like(component_mask)
    cv2.ellipse(ellipse_mask, ellipse, 255, -1)  # Draw the fitted ellipse on the output image to visually verify the bean shape.

    # Intersection over union measures how closely the shapes overlap.
    intersection = np.count_nonzero(cv2.bitwise_and(filled_component, ellipse_mask))  # Count the selected pixels for area or coverage calculations.
    union = np.count_nonzero(cv2.bitwise_or(filled_component, ellipse_mask))  # Count the selected pixels for area or coverage calculations.
    ellipse_iou = intersection / union if union > 0 else 0.0  

    # Compare the component area with the ellipse area.
    component_area = np.count_nonzero(filled_component)  # Count the selected pixels for area or coverage calculations.
    ellipse_area = np.count_nonzero(ellipse_mask)  # Count the selected pixels for area or coverage calculations.
    area_ratio = min(component_area, ellipse_area) / max(component_area, ellipse_area) if ellipse_area > 0 else 0.0  # Compare the bean area with the fitted ellipse area to check how well the ellipse matches the shape.

    # Calculate the fitted ellipse's major-to-minor axis ratio.
    (_, _), (axis_1, axis_2), _ = ellipse
    major_axis = max(axis_1, axis_2)  # Compute the long axis of the fitted ellipse so the bean elongation can be evaluated.
    minor_axis = min(axis_1, axis_2)  # Compute the short axis so the aspect ratio can be compared to natural bean shapes.
    axis_ratio = major_axis / minor_axis if minor_axis > 0 else 0.0  

    return ellipse_iou, area_ratio, axis_ratio, ellipse


def calculate_score(area: float, solidity: float, circularity: float, ellipse_iou: float, ellipse_area_ratio: float, axis_ratio: float, min_area: int, max_area: int) -> float:  # Combine the geometric metrics into a single bean-likeness score.
    """Combine the shape properties into one potential-bean score."""  
    area_score = 1.0 if min_area <= area <= max_area else 0.0  
    axis_score = 1.0 if 1.0 <= axis_ratio <= 2.5 else 0.0  

    return (
        0.10 * area_score  
        + 0.10 * axis_score  
        + 0.20 * solidity  
        + 0.10 * circularity  
        + 0.35 * ellipse_iou  
        + 0.15 * ellipse_area_ratio  
    )  


def analyse_components(binary: np.ndarray, min_area: int, max_area: int, min_solidity: float, min_iou: float, min_area_ratio: float, min_axis_ratio: float, max_axis_ratio: float, min_score: float):  # Measure each connected bean candidate and rank the most bean-like shapes.
    """Measure every connected component and rank bean-like shapes."""  
    number_of_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)  # Label each connected region so the script can inspect them one at a time.

    output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)  # Convert the image to a different colour space.
    results = []

    for label in range(1, number_of_labels):  # Iterate over every item in the collection so each region is processed individually.
        x = stats[label, cv2.CC_STAT_LEFT]  
        y = stats[label, cv2.CC_STAT_TOP]  
        width = stats[label, cv2.CC_STAT_WIDTH]  
        height = stats[label, cv2.CC_STAT_HEIGHT]  
        area = stats[label, cv2.CC_STAT_AREA]  
        centre_x, centre_y = centroids[label]  

        component_mask = np.zeros_like(binary)
        component_mask[labels == label] = 255

        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # Find the outlines of each bean-shaped region so they can be measured individually.

        if not contours:
            continue  

        contour = max(contours, key=cv2.contourArea)  
        perimeter = cv2.arcLength(contour, True)  # Measure the contour length so the circularity metric can be calculated for the component.

        hull = cv2.convexHull(contour)  # Build the convex hull so the script can estimate solidity and shape irregularity.
        hull_area = cv2.contourArea(hull)  # Measure the contour area so the script can compare shape and size against bean thresholds.

        solidity = area / hull_area if hull_area > 0 else 0.0  
        circularity = calculate_circularity(area, perimeter)  

        ellipse_iou, ellipse_area_ratio, ellipse_axis_ratio, ellipse = calculate_ellipse_match(component_mask, contour)  

        score = calculate_score(  
            area,  
            solidity,  
            circularity,  
            ellipse_iou,  
            ellipse_area_ratio,  
            ellipse_axis_ratio,  
            min_area,  
            max_area,  
        )  

        is_candidate = (  
            min_area <= area <= max_area  
            and solidity >= min_solidity  
            and ellipse_iou >= min_iou  
            and ellipse_area_ratio >= min_area_ratio  
            and min_axis_ratio <= ellipse_axis_ratio <= max_axis_ratio  
            and score >= min_score  
        )  

        colour = (0, 255, 0) if is_candidate else (0, 0, 255)  

        cv2.drawContours(output, [contour], -1, colour, 2)  # Draw the bean boundary onto the output image to highlight the detected component.
        cv2.rectangle(output, (x, y), (x + width, y + height), colour, 1)  

        if ellipse is not None:
            ellipse_colour = (255, 255, 0) if is_candidate else (100, 100, 100)  
            cv2.ellipse(output, ellipse, ellipse_colour, 2)  # Draw the fitted ellipse on the output image to visually verify the bean shape.

        cv2.putText(output, str(label), (int(centre_x) - 8, int(centre_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

        results.append({  
            "label": label,  
            "area": area,  
            "solidity": solidity,  
            "circularity": circularity,  
            "ellipse_iou": ellipse_iou,  
            "ellipse_area_ratio": ellipse_area_ratio,  
            "ellipse_axis_ratio": ellipse_axis_ratio,  
            "score": score,  
            "candidate": is_candidate,  
        })  

    return output, results


def add_information(image: np.ndarray, results: list[dict], selected_count: int, min_area: int, max_area: int, min_solidity: float, min_iou: float, min_area_ratio: float, min_axis_ratio: float, max_axis_ratio: float, min_score: float) -> np.ndarray:
    """Show the active filter values and candidate counts."""  
    output = image.copy()  

    candidate_count = sum(result["candidate"] for result in results)  

    lines = [  # Store the fixed reference values used throughout this analysis.
        f"Components: {len(results)}",  
        f"Passing filters: {candidate_count}",  
        f"Selected best N: {selected_count}",  
        f"Area: {min_area} to {max_area}",  
        f"Minimum solidity: {min_solidity:.2f}",  
        f"Minimum ellipse IoU: {min_iou:.2f}",  
        f"Minimum ellipse area ratio: {min_area_ratio:.2f}",  
        f"Ellipse axis ratio: {min_axis_ratio:.2f} to {max_axis_ratio:.2f}",  
        f"Minimum score: {min_score:.2f}",  
        "Green = selected bean",  
        "Red = rejected",  
        "Yellow = fitted ellipse",  
        "Press S to save",  
    ]  

    for index, text in enumerate(lines):  # Iterate over every item in the collection so each region is processed individually.
        cv2.putText(output, text, (15, 28 + index * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 0), 2)

    return output


def select_best_candidates(analysed: np.ndarray, results: list[dict], labels: np.ndarray, number_to_select: int) -> tuple[np.ndarray, list[dict]]:
    """Keep only the best-scoring components that already passed the filters."""  
    selected_output = analysed.copy()  

    passing = [result for result in results if result["candidate"]]
    passing.sort(key=lambda result: result["score"], reverse=True)  
    selected = passing[:number_to_select]  

    selected_labels = {result["label"] for result in selected}

    # Redraw selected components with thicker green outlines.
    for result in results:  # Iterate over every item in the collection so each region is processed individually.
        if result["label"] not in selected_labels:
            continue  

        component_mask = np.zeros(labels.shape, dtype=np.uint8)  
        component_mask[labels == result["label"]] = 255

        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # Find the outlines of each bean-shaped region so they can be measured individually.

        if contours:
            contour = max(contours, key=cv2.contourArea)  
            cv2.drawContours(selected_output, [contour], -1, (0, 255, 0), 4)  # Draw the bean boundary onto the output image to highlight the detected component.

    return selected_output, selected


def main() -> None:  # Run the full Otsu threshold experiment over the Lab channels.
    image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)  # Load an image from disk using OpenCV.

    if image is None:
        raise FileNotFoundError(f"Could not open image: {IMAGE_PATH}")

    # Convert to strict binary and invert because beans are black in the input.
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)  # Threshold the image to create a binary mask.
    binary = cv2.bitwise_not(binary)  # Invert the mask when the user wants dark beans treated as the foreground.

    _, labels, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)  # Label each connected region so the script can score and rank bean-like components.

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)  
    cv2.resizeWindow(WINDOW, 1500, 900)

    cv2.createTrackbar("Minimum area", WINDOW, 4500, 30000, do_nothing)  # Add an interactive slider so the user can tune the current processing parameter live.
    cv2.createTrackbar("Maximum area", WINDOW, 29000, 60000, do_nothing)  # Add an interactive slider so the user can tune the current processing parameter live.
    cv2.createTrackbar("Minimum solidity x100", WINDOW, 60, 100, do_nothing)  # Add an interactive slider so the user can tune the current processing parameter live.
    cv2.createTrackbar("Minimum IoU x100", WINDOW, 60, 100, do_nothing)  # Add an interactive slider so the user can tune the current processing parameter live.
    cv2.createTrackbar("Minimum area ratio x100", WINDOW, 70, 100, do_nothing)  # Add an interactive slider so the user can tune the current processing parameter live.
    cv2.createTrackbar("Minimum axis ratio x100", WINDOW, 100, 300, do_nothing)  # Add an interactive slider so the user can tune the current processing parameter live.
    cv2.createTrackbar("Maximum axis ratio x100", WINDOW, 250, 500, do_nothing)  # Add an interactive slider so the user can tune the current processing parameter live.
    cv2.createTrackbar("Minimum score x100", WINDOW, 65, 100, do_nothing)  # Add an interactive slider so the user can tune the current processing parameter live.
    cv2.createTrackbar("Select best N", WINDOW, 5, 20, do_nothing)  # Add an interactive slider so the user can tune the current processing parameter live.

    while True:  # Keep the preview loop running until the user exits the window.
        min_area = cv2.getTrackbarPos("Minimum area", WINDOW)  # Read the current slider value so the live preview matches the chosen settings.
        max_area = max(cv2.getTrackbarPos("Maximum area", WINDOW), min_area + 1)  # Read the current slider value so the live preview matches the chosen settings.
        min_solidity = cv2.getTrackbarPos("Minimum solidity x100", WINDOW) / 100.0  # Read the current slider value so the live preview matches the chosen settings.
        min_iou = cv2.getTrackbarPos("Minimum IoU x100", WINDOW) / 100.0  # Read the current slider value so the live preview matches the chosen settings.
        min_area_ratio = cv2.getTrackbarPos("Minimum area ratio x100", WINDOW) / 100.0  # Read the current slider value so the live preview matches the chosen settings.
        min_axis_ratio = max(cv2.getTrackbarPos("Minimum axis ratio x100", WINDOW), 1) / 100.0  # Read the current slider value so the live preview matches the chosen settings.
        max_axis_ratio = max(cv2.getTrackbarPos("Maximum axis ratio x100", WINDOW), 1) / 100.0  # Read the current slider value so the live preview matches the chosen settings.
        min_score = cv2.getTrackbarPos("Minimum score x100", WINDOW) / 100.0  # Read the current slider value so the live preview matches the chosen settings.
        number_to_select = max(cv2.getTrackbarPos("Select best N", WINDOW), 1)  # Read the current slider value so the live preview matches the chosen settings.

        analysed, results = analyse_components(  
            binary,  
            min_area,  
            max_area,  
            min_solidity,  
            min_iou,  
            min_area_ratio,  
            min_axis_ratio,  
            max_axis_ratio,  
            min_score,  
        )  

        analysed, selected = select_best_candidates(analysed, results, labels, number_to_select)  

        analysed = add_information(  # Overlay the current operation settings on the processed image for easier interpretation.
            analysed,  
            results,  
            len(selected),  
            min_area,  
            max_area,  
            min_solidity,  
            min_iou,  
            min_area_ratio,  
            min_axis_ratio,  
            max_axis_ratio,  
            min_score,  
        )  

        display = cv2.resize(analysed, (1000, 800), interpolation=cv2.INTER_NEAREST)
        cv2.imshow(WINDOW, display)  # Display the current result in the OpenCV window so the user can inspect it live.

        key = cv2.waitKey(20) & 0xFF  

        if key in (ord("q"), 27):
            break  

        if key == ord("s"):
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)  

            cv2.imwrite(str(RESULTS_DIR / "selected_bean_candidates.png"), analysed)  # Save the current image to a file.
            cv2.imwrite(str(RESULTS_DIR / "component_binary.png"), binary)  # Save the current image to a file.

            csv_path = RESULTS_DIR / "component_measurements.csv"  

            with csv_path.open("w", encoding="utf-8") as file:  
                file.write("label,area,solidity,circularity,ellipse_iou,ellipse_area_ratio,ellipse_axis_ratio,score,candidate\n")  

                for result in sorted(results, key=lambda item: item["score"], reverse=True):  # Iterate over every item in the collection so each region is processed individually.
                    file.write(  
                        f"{result['label']},"  
                        f"{result['area']},"  
                        f"{result['solidity']:.4f},"  
                        f"{result['circularity']:.4f},"  
                        f"{result['ellipse_iou']:.4f},"  
                        f"{result['ellipse_area_ratio']:.4f},"  
                        f"{result['ellipse_axis_ratio']:.4f},"  
                        f"{result['score']:.4f},"  
                        f"{result['candidate']}\n"  
                    )  

            print("\nSelected components:")  # Print the current result to the console.
            for rank, result in enumerate(selected, start=1):  # Iterate over every item in the collection so each region is processed individually.
                print(f"{rank}. Component {result['label']}: score={result['score']:.3f}, IoU={result['ellipse_iou']:.3f}")  # Print the current result to the console.

            print(f"\nResults saved in: {RESULTS_DIR}")  # Print the current result to the console.

    cv2.destroyAllWindows()  


if __name__ == "__main__":  # Run the script only when this file is executed directly.
    main()  

