import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os

def compute_metrics(original, processed):
    # Convert images to grayscale
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    # Compute SSIM
    similarity_index = ssim(original_gray, processed_gray)

    # Compute PSNR
    mse = np.mean((original_gray - processed_gray) ** 2)
    psnr_value = 10 * np.log10(255**2 / mse) if mse != 0 else float('inf')

    return similarity_index, psnr_value

def compute_miou(ground_truth, prediction):
    intersection = np.logical_and(ground_truth, prediction).sum()
    union = np.logical_or(ground_truth, prediction).sum()
    return intersection / union if union != 0 else 0

def edge_detection(image_path, save_results=True):
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not load image. Check the file path.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply different edge detection techniques
    edges_canny = cv2.Canny(gray, 100, 200)
    edges_sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    edges_laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Convert Sobel and Laplacian to uint8 format for visualization
    edges_sobel = cv2.convertScaleAbs(edges_sobel)
    edges_laplacian = cv2.convertScaleAbs(edges_laplacian)

    # Compute performance metrics
    ssim_value, psnr_value = compute_metrics(image, cv2.cvtColor(edges_canny, cv2.COLOR_GRAY2BGR))
    miou_value = compute_miou(gray > 128, edges_canny > 128)

    # Display images
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(edges_canny, cmap="gray")
    plt.title("Canny Edge Detection")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(edges_sobel, cmap="gray")
    plt.title("Sobel Edge Detection")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(edges_laplacian, cmap="gray")
    plt.title("Laplacian Edge Detection")
    plt.axis("off")

    plt.suptitle(f"SSIM: {ssim_value:.4f} | PSNR: {psnr_value:.2f} dB | mIoU: {miou_value:.4f}")
    plt.show()

    if save_results:
        output_folder = "output_images"
        os.makedirs(output_folder, exist_ok=True)
        cv2.imwrite(os.path.join(output_folder, "canny_edges.png"), edges_canny)
        cv2.imwrite(os.path.join(output_folder, "sobel_edges.png"), edges_sobel)
        cv2.imwrite(os.path.join(output_folder, "laplacian_edges.png"), edges_laplacian)
        print("Edge detection results saved successfully.")

# Run the function
edge_detection("input.jpg")