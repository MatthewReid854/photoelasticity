# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# photoelasticity: A Python library to visualise surface strain using PVST.
#
# Copyright (c) 2025 Matthew Reid
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------
# Author:   Matthew Reid
# Contact:  alpha.reliability@gmail.com
# Repo:     https://github.com/MatthewReid854/photoelasticity
# -----------------------------------------------------------------------------


import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path


def process_polarized_image(input_image_path, directory_name, black_image_path=None, show_plot=True):
    """
    Processes a polarized image with 4x4 superpixels containing four
    2x2 polarization sub-pixels (0, 45, 90, 135 degrees).

    It averages each 2x2 sub-pixel region, splits the image into four
    separate images based on polarization, and saves them.

    Args:
      input_image_path (str): Path to the input polarized image file.
      directory_name (str): Directory where the four output polarization images will be saved.
      black_image_path (str): Path to the black image (taken with lens cap on to measure sensor noise). Optional.
      show_plot (bool): If True, the plot of all 4 processed images will be shown.
    """

    # Load and Subtract Images using OpenCV
    print(f"Processing image: {input_image_path}")
    standard_img = cv2.imread(input_image_path)

    # Check if the standard image was loaded
    if standard_img is None:
        print(f"Error: Could not load the input image at {input_image_path}")
        return

    if black_image_path is not None:
        black_img = cv2.imread(black_image_path)
        if black_img is None:
            print(f"Warning: Could not load black image at {black_image_path}. Proceeding without subtraction.")
            img = standard_img
        else:
            # Subtract the black image from the standard image
            img = cv2.subtract(standard_img, black_img)
    else:
        img = standard_img

    # Get image dimensions from the NumPy array's shape.
    image_height, image_width, _ = img.shape

    # Crop the image using NumPy array slicing.
    img_cropped = img[1:image_height - 3, 1:image_width - 3]

    # Convert the BGR image to grayscale.
    img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)

    # Get dimensions of the grayscale image array
    height, width = img_gray.shape

    # Check if dimensions are divisible by 4 (required for 4x4 superpixels)
    if height % 4 != 0 or width % 4 != 0:
        print(f"Error: Image dimensions after cropping ({width}x{height}) must be divisible by 4.")
        return

    # Calculate output dimensions
    out_height = height // 4
    out_width = width // 4
    print(f"Input image dimensions: {image_width}x{image_height}, Output image dimensions: {out_width}x{out_height}")

    # Create Output Arrays
    pol0_img = np.zeros((out_height, out_width), dtype=np.float64)
    pol45_img = np.zeros((out_height, out_width), dtype=np.float64)
    pol90_img = np.zeros((out_height, out_width), dtype=np.float64)
    pol135_img = np.zeros((out_height, out_width), dtype=np.float64)

    # Process Superpixels
    for i in range(0, height, 4):
        for j in range(0, width, 4):
            out_y = i // 4
            out_x = j // 4

            # Extract sub-regions from the corrected grayscale numpy array
            pol0_sub = img_gray[i + 2:i + 4, j + 2:j + 4]
            pol45_sub = img_gray[i:i + 2, j + 2:j + 4]
            pol90_sub = img_gray[i:i + 2, j:j + 2]
            pol135_sub = img_gray[i + 2:i + 4, j:j + 2]

            # Calculate the average and assign to output arrays
            pol0_img[out_y, out_x] = np.mean(pol0_sub)
            pol45_img[out_y, out_x] = np.mean(pol45_sub)
            pol90_img[out_y, out_x] = np.mean(pol90_sub)
            pol135_img[out_y, out_x] = np.mean(pol135_sub)

    # Save and Display
    current_directory = Path.cwd()
    subfolder_path = current_directory / directory_name
    try:
        subfolder_path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"Creation of the directory {subfolder_path} failed: {error}")
        return

    # Round the float values to the nearest integer for better precision.
    pol0_rounded = np.round(pol0_img)
    pol45_rounded = np.round(pol45_img)
    pol90_rounded = np.round(pol90_img)
    pol135_rounded = np.round(pol135_img)

    # Scale the 8-bit data to the full 16-bit range. Multiplying by 257 correctly maps 0->0 and 255->65535.
    pol0_scaled = (pol0_rounded * 257).astype(np.uint16)
    pol45_scaled = (pol45_rounded * 257).astype(np.uint16)
    pol90_scaled = (pol90_rounded * 257).astype(np.uint16)
    pol135_scaled = (pol135_rounded * 257).astype(np.uint16)

    # Define output paths.
    base_name = os.path.basename(input_image_path)
    name, _ = os.path.splitext(base_name)
    output_paths = {
        'pol0': os.path.join(subfolder_path, f"{name}_pol0.tiff"),
        'pol45': os.path.join(subfolder_path, f"{name}_pol45.tiff"),
        'pol90': os.path.join(subfolder_path, f"{name}_pol90.tiff"),
        'pol135': os.path.join(subfolder_path, f"{name}_pol135.tiff"),
    }

    # Save the properly scaled 16-bit images using OpenCV.
    cv2.imwrite(output_paths['pol0'], pol0_scaled)
    cv2.imwrite(output_paths['pol45'], pol45_scaled)
    cv2.imwrite(output_paths['pol90'], pol90_scaled)
    cv2.imwrite(output_paths['pol135'], pol135_scaled)

    print(f"Successfully saved 4 scaled 16-bit polarization images to '{directory_name}'")

    # Plotting for visualization. We use the 'rounded' 8-bit range data for correct display contrast.
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    axs[0, 1].imshow(pol45_rounded, cmap='gray', vmin=0, vmax=255)
    axs[0, 1].set_title('Polarization 45°')
    axs[0, 1].axis('off')
    axs[0, 0].imshow(pol90_rounded, cmap='gray', vmin=0, vmax=255)
    axs[0, 0].set_title('Polarization 90°')
    axs[0, 0].axis('off')
    axs[1, 0].imshow(pol135_rounded, cmap='gray', vmin=0, vmax=255)
    axs[1, 0].set_title('Polarization 135°')
    axs[1, 0].axis('off')
    axs[1, 1].imshow(pol0_rounded, cmap='gray', vmin=0, vmax=255)
    axs[1, 1].set_title('Polarization 0°')
    axs[1, 1].axis('off')

    fig.suptitle(f'Polarization Channels for {base_name}', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(subfolder_path / (name + '_pol_combined.png'), bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
