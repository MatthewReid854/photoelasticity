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
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from PIL import Image
from pathlib import Path


def get_alignment_points(directory_name, image_name):
    """Interactively displays an image to let a user select four alignment points.

    This function loads a specified image from a subfolder and displays it in
    a Matplotlib window. It overlays four distinct, colored crosshairs that
    the user can move by typing coordinates into corresponding text boxes.

    The primary purpose is to allow for the manual and precise selection of
    four reference points (e.g., corners of a region of interest) for
    subsequent image processing or analysis.

    Parameters
    ----------
    directory_name : str
        The name of the subfolder where the target image is located.
    image_name : str
        The base name of the image file. The function will append '_pol0.tiff' to this name to find the file.

    Returns
    -------
    None
        This function does not return any value. Its main output is the interactive Matplotlib window, which stays
        open until the user manually closes it. Once you have the crosshair coordinates, please record them for use in
        the transform_images function.

    """

    # Configuration and data
    COLORS = ['red', 'blue', 'green', 'gold']
    NAMES = ['Top Left', 'Top Right', 'Bottom Left', 'Bottom Right']
    NUM_CROSSHAIRS = 4

    # Create the Path object for the new subfolder
    current_directory = Path.cwd()
    subfolder_path = current_directory / directory_name
    input_image_path = subfolder_path / (image_name + '_pol0.tiff')

    # Load the image
    img = Image.open(input_image_path)
    width, height = img.size
    img_data = img.convert('L')  # Convert to grayscale ('L' = luminosity) for intensity averaging

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle('Image with 4 Interactive Crosshairs')

    # Adjust plot to make room for the widgets on the right
    plt.subplots_adjust(right=0.72, bottom=0.1)

    # Display the image and set axis properties
    ax.imshow(img_data, cmap='gray')
    ax.set_title('Update coordinates in the text boxes')
    ax.set_ylim(height - 1, 0)
    ax.set_xlim(0, width - 1)

    # Define the interactive update function
    def submit(index):
        """
        Updates the position of the crosshair specified by 'index'.
        This function is called by the event handler for the text boxes.
        """
        # Get the correct pair of text boxes using the index
        tb_x, tb_y = text_boxes[index]
        # Get the correct line objects using the index
        lines = crosshair_lines[index]

        try:
            x_coord = float(tb_x.text)
            y_coord = float(tb_y.text)

            # Update the data for the vertical and horizontal lines
            lines['vline'].set_data([x_coord, x_coord], [y_coord - line_length / 2, y_coord + line_length / 2])
            lines['hline'].set_data([x_coord - line_length / 2, x_coord + line_length / 2], [y_coord, y_coord])

            # Redraw the plot canvas to show the changes
            fig.canvas.draw_idle()

        except ValueError:
            print(f"Invalid input for Crosshair {index + 1}. Please use numbers.")

    # create crosshairs and textboxes
    crosshair_lines = []
    text_boxes = []
    line_length = 0.10 * min(width, height)

    # Loop to create each set of crosshairs and its UI controls
    for i in range(NUM_CROSSHAIRS):
        # Stagger initial positions for better visibility
        if i == 0:
            initial_x = width * 0.2
            initial_y = height * 0.2
        if i == 1:
            initial_x = width * 0.8
            initial_y = height * 0.2
        if i == 2:
            initial_x = width * 0.2
            initial_y = height * 0.8
        if i == 3:
            initial_x = width * 0.8
            initial_y = height * 0.8

        # Create and store the crosshair lines using ax.plot()
        v_line, = ax.plot([initial_x, initial_x], [initial_y - line_length / 2, initial_y + line_length / 2],
                          color=COLORS[i], lw=1, linestyle='-')
        h_line, = ax.plot([initial_x - line_length / 2, initial_x + line_length / 2], [initial_y, initial_y],
                          color=COLORS[i], lw=1, linestyle='-')
        crosshair_lines.append({'vline': v_line, 'hline': h_line})

        # Create the text boxes for the current crosshair
        # Calculate y-position for this set of controls
        y_pos = 0.85 - (i * 0.17)
        # Define the axes for the text boxes [left, bottom, width, height]
        ax_box_x = plt.axes([0.77, y_pos, 0.18, 0.05])
        ax_box_y = plt.axes([0.77, y_pos - 0.06, 0.18, 0.05])

        # Add a title for this crosshair group
        ax_box_x.text(-0.1, 1.7, 'Crosshair ' + NAMES[i], transform=ax_box_x.transAxes,
                      verticalalignment='top', color=COLORS[i], weight='bold')

        # Create the TextBox widgets
        text_box_x = TextBox(ax_box_x, 'X-Coord', initial=f"{initial_x:.1f}")
        text_box_y = TextBox(ax_box_y, 'Y-Coord', initial=f"{initial_y:.1f}")

        # Connect the 'on_submit' event to the 'submit' function.
        text_box_x.on_submit(lambda text, index=i: submit(index))
        text_box_y.on_submit(lambda text, index=i: submit(index))

        # Store the text box objects for later access
        text_boxes.append((text_box_x, text_box_y))

    # Display the plot
    plt.show()


def register_image_perspective(reference_image, loaded_image, reference_points, loaded_points):
    """
    Performs a perspective transformation to register a loaded image to a reference image.

    This function calculates the homography matrix required to align the
    control points from the loaded image with those in the reference image,
    and then applies this transformation to the entire loaded image.

    Args:
        reference_image (np.ndarray): The target image for alignment.
        loaded_image (np.ndarray): The image to be transformed.
        reference_points (list or np.ndarray): A list of 4 or more (x, y) coordinates of control points in the reference image.
        loaded_points (list or np.ndarray): A list of the corresponding 4 or more (x, y) coordinates in the loaded image.

    Returns:
        np.ndarray: The loaded image, transformed to be aligned with the reference image.
                    Returns None if input is invalid.
    """
    # Input Validation
    if len(reference_points) < 4 or len(loaded_points) < 4:
        print("Error: At least four corresponding points are required for perspective transformation.")
        return None

    # Convert points to NumPy arrays of float32
    src_pts = np.float32(loaded_points).reshape(-1, 1, 2)
    dst_pts = np.float32(reference_points).reshape(-1, 1, 2)

    # Calculate the Perspective Transformation Matrix (Homography)
    # Use findHomography which can use 4+ points for robust estimation
    transformation_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if transformation_matrix is None:
        print("Error: Could not compute the homography matrix.")
        return None

    # Apply the Transformation to the Loaded Image
    h, w = reference_image.shape[:2]
    registered_image = cv2.warpPerspective(loaded_image, transformation_matrix, (w, h))

    return registered_image


def register_image_affine(reference_image, loaded_image, reference_points, loaded_points):
    """
    Performs affine transformation to register a loaded image to a reference image.

    This function calculates the transformation matrix required to align the
    control points from the loaded image with those in the reference image,
    and then applies this transformation to the entire loaded image.

    Args:
        reference_image (np.ndarray): The target image for alignment.
        loaded_image (np.ndarray): The image to be transformed.
        reference_points (list or np.ndarray): A list of 3 (x, y) coordinates of control points in the reference image.
        loaded_points (list or np.ndarray): A list of the corresponding 3 (x, y) coordinates in the loaded image.

    Returns:
        np.ndarray: The loaded image, transformed to be aligned with the reference image. Returns None if input is invalid.
    """
    # Input Validation
    if len(reference_points) < 3 or len(loaded_points) < 3:
        print("Error: At least three corresponding points are required for affine transformation.")
        return None
    if len(reference_points) != len(loaded_points):
        print("Error: The number of reference points and loaded points must be equal.")
        return None

    # Convert points to NumPy arrays of float32
    # getAffineTransform requires points in a (3, 2) shape
    src_pts = np.float32(loaded_points)[:3]
    dst_pts = np.float32(reference_points)[:3]

    # Calculate the Affine Transformation Matrix
    transformation_matrix = cv2.getAffineTransform(src_pts, dst_pts)

    # Apply the Transformation to the Loaded Image
    h, w = reference_image.shape[:2]
    registered_image = cv2.warpAffine(loaded_image, transformation_matrix, (w, h))

    return registered_image


def transform_images(directory_name, unstrained_image_name, strained_image_name, unstrained_points, strained_points, show_plot=None):
    """Aligns strained images to unstrained images using perspective transformation.

    This function processes image sets from four polarisations (0°, 45°, 90°, 135°). For each polarisation, it uses
    corresponding points from the unstrained and strained images to calculate a transformation matrix. It then warps
    the strained image to align it with the unstrained reference image.

    The newly created "corrected" images are saved to the disk. Finally, a summary plot is generated and saved, showing
    the original unstrained, original strained, and an overlay of the unstrained and corrected images to visually assess
    the quality of the alignment.

    Parameters
    ----------
    directory_name : str
        The name of the subfolder containing all image files.
    unstrained_image_name : str
        The base name for the unstrained (reference) image set.
    strained_image_name : str
        The base name for the strained image set that will be transformed.
    unstrained_points : array-like
        A list or array of four [x, y] coordinates from the unstrained images.
    strained_points : array-like
        The corresponding list or array of four [x, y] coordinates from the strained images.
    show_plot : bool, optional
        If True, the summary plot will be displayed in an interactive window.
        If False or None (default), the plot is saved and the figure is closed.

    Returns
    -------
    None

    Side Effects
    ------------
    - Saves four corrected images to disk with the filename format:
      `{strained_image_name}_CORRECTED_{polarisation}.tiff`.
    - Saves a summary plot to disk named `image_transformation_plot.png`.
    """

    current_directory = Path.cwd()
    subfolder_path = current_directory / directory_name

    # Lists to store images for plotting later
    polarisations_list = ['pol0', 'pol45', 'pol90', 'pol135']
    unstrained_images = []
    strained_images = []
    corrected_images = []

    for polarisation in polarisations_list:
        print(f"--- Processing Polarisation: {polarisation} ---")
        unstrained_image_path = subfolder_path / f'{unstrained_image_name}_{polarisation}.tiff'
        unstrained_image = np.array(Image.open(unstrained_image_path))

        strained_image_path = subfolder_path / f'{strained_image_name}_{polarisation}.tiff'
        strained_image = np.array(Image.open(strained_image_path))

        # Run the registration function
        corrected_image = register_image_perspective(
            unstrained_image,
            strained_image,
            unstrained_points,
            strained_points
        )

        # Store images for plotting
        unstrained_images.append(unstrained_image)
        strained_images.append(strained_image)
        corrected_images.append(corrected_image)

        # Check if registration was successful before saving
        if corrected_image is not None:
            output_path = subfolder_path / f"{strained_image_name}_CORRECTED_{polarisation}.tiff"
            corrected_image_pil = Image.fromarray(corrected_image.astype(np.uint8), 'L')
            corrected_image_pil.save(output_path)
        else:
            print(f"Skipping save for {polarisation} due to registration failure.")

    print(f"\nImage correction complete")

    # Plotting Section
    print("\n--- Generating Plots ---")
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    fig.suptitle('Image Registration Results', fontsize=24, y=0.95)

    # Create a map for display titles
    title_map = {
        'pol0': 'Polarisation 0°',
        'pol45': 'Polarisation 45°',
        'pol90': 'Polarisation 90°',
        'pol135': 'Polarisation 135°'
    }

    for i, pol in enumerate(polarisations_list):
        # Get the images for the current column
        unstrained = unstrained_images[i]
        strained = strained_images[i]
        corrected = corrected_images[i]

        # Column Title
        display_title = title_map.get(pol, pol)  # Get the formatted title from the map
        axes[0, i].set_title(display_title, fontsize=16, pad=10)

        # Row 0: Unstrained (Reference) Image
        axes[0, i].imshow(unstrained, cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].text(-0.1, 0.5, 'Unstrained\n(Reference)', transform=axes[0, i].transAxes,
                            ha='center', va='center', rotation=90, fontsize=14)

        # Row 1: Strained (Original) Image
        axes[1, i].imshow(strained, cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].text(-0.1, 0.5, 'Strained\n(Original)', transform=axes[1, i].transAxes,
                            ha='center', va='center', rotation=90, fontsize=14)

        # Row 2: Overlay of Unstrained and Corrected
        if corrected is not None:
            # Create a color overlay to visualize alignment
            # Unstrained (reference) in Red channel, Corrected in Green channel
            # Well-aligned areas will appear Yellow
            overlay = np.zeros((unstrained.shape[0], unstrained.shape[1], 3), dtype=np.uint8)
            overlay[..., 0] = cv2.normalize(unstrained, None, 0, 255, cv2.NORM_MINMAX)  # Red channel
            overlay[..., 1] = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)  # Green channel
            axes[2, i].imshow(overlay)
        else:
            # If correction failed, show a black image with text
            axes[2, i].set_facecolor('black')
            axes[2, i].text(0.5, 0.5, 'Correction\nFailed', color='white',
                            ha='center', va='center', transform=axes[2, i].transAxes)

        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].text(-0.1, 0.5, 'Overlay\n(Unstrained+Corrected)', transform=axes[2, i].transAxes,
                            ha='center', va='center', rotation=90, fontsize=14)

    plt.tight_layout(rect=[0.05, 0, 1, 0.95])  # Adjust layout to make room for suptitle and row labels
    plt.savefig(subfolder_path / 'image_transformation_plot.png', bbox_inches='tight')

    if show_plot is True:
        plt.show()
    else:
        plt.close(fig)
