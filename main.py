"""
Image Processing & Foreground Segmentation Pipeline
---------------------------------------------------
Author: Yash
Description:
This script performs a complete image enhancement and 
foreground segmentation pipeline using custom filters
and region growing.

Pipeline Steps:
1. Image Resize & Color Conversion
2. Sharpening & Smoothing
3. Edge Detection
4. Binary Thresholding
5. Morphological Cleaning
6. Region Growing Segmentation
7. Foreground Extraction
"""

# ================= IMPORTS =================
import cv2
import time
import matplotlib.pyplot as plt

from filters.processing import (
    operation,
    threshold,
    threshold2,
    foreground,
    binary_smooth,
    binary_dilate,
    binary_erode,
    connect_center,
    background_remove
)

# ================= MAIN FUNCTION =================
def main():

    start_time = time.time()

    # -------- Load Image --------
    image_path = "input/1 (95).jpg"
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # -------- Preprocessing --------
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (200, 200))
    img_copy = img.copy()

    # -------- Image Processing --------
    binary_mask = threshold(img, 200)
    smoothed_binary = binary_smooth(binary_mask)
    seed_mask = threshold2(img_copy, 100)

    sharpened_img = operation(img, "sharp")
    
    smoothed_img = operation(img, "smooth")

    edge_x = operation(img, "edge1")
    edge_y = operation(img, "edge2")
    edge_map = edge_x + edge_y

    cleaned_mask = background_remove(binary_mask, 200)

    # -------- Region Growing --------
    final_mask = connect_center(cleaned_mask.copy(), seed_mask.copy())

    foreground_masked = foreground(img, final_mask, 0)
    final_foreground = foreground(img, final_mask, 1)

    # ================= VISUALIZATION =================
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    fig.suptitle("Image Processing & Segmentation Pipeline",
                 fontsize=16, fontweight="bold")

    images = [
        (img, "Original Image", None),
        (sharpened_img, "Sharpened Image", None),
        (smoothed_img, "Smoothed Image", None),
        (edge_map, "Edge Detection (Sobel)", None),
        (binary_mask, "Binary Threshold", "gray"),
        (seed_mask, "Initial Seed Mask", "gray"),
        (foreground_masked, "Region-Grown Mask", None),
        (final_foreground, "Final Foreground Extraction", None)
    ]

    for ax, (image, title, cmap) in zip(axes.flatten(), images):
        if cmap:
            ax.imshow(image, cmap=cmap)
        else:
            ax.imshow(image)
        ax.set_title(title, fontsize=10, pad=6)
        ax.axis("off")

    # Proper spacing adjustment
    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        bottom=0.07,
        top=0.90,
        wspace=0.20,
        hspace=0.35
    )

    plt.savefig("results.png", dpi=300)
    plt.show()

    # ================= REPORT =================
    total_pixels = img.shape[0] * img.shape[1]
    foreground_pixels = int(final_mask.sum())
    coverage = (foreground_pixels / total_pixels) * 100
    processing_time = time.time() - start_time

    print("\n" + "="*60)
    print(" IMAGE SEGMENTATION PIPELINE REPORT")
    print("="*60)

    print(f"• Image Size                : {img.shape[0]} x {img.shape[1]} pixels")
    print(f"• Total Pixels              : {total_pixels}")
    print(f"• Foreground Pixels Detected: {foreground_pixels}")
    print(f"• Foreground Coverage       : {coverage:.2f}%")
    print(f"• Processing Time           : {processing_time:.4f} seconds")

    print("\n✔ Segmentation Completed Successfully")
    print("="*60 + "\n")


# ================= ENTRY POINT =================
if __name__ == "__main__":
    main()
