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


from reliability.Other_functions import histogram
import matplotlib.patches as patches
from reliability.Fitters import Fit_Normal_2P, Fit_Weibull_Mixture, Fit_Gamma_3P
from scipy.stats import linregress
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from pathlib import Path
from matplotlib.gridspec import GridSpec
import cv2
from matplotlib.colors import LogNorm


def create_masked_amplitude_array(directory_name, strained_image_name, bounding_box):
    """ Identifies enclosed features in an image and masks the corresponding data points in an amplitude array by
    setting them to NaN.

    This function uses a contour hierarchy method with a size filter to identify enclosed "holes" within a given
    bounding box.

    Parameters
    ----------
    directory_name (str):
        The name of the directory containing the data files.
    strained_image_name (str):
        The base name of the strained image files.
    bounding_box (tuple):
        A tuple defining the ROI as (row_start, row_end, column_start, column_end).

    Returns
    -------
    None
    """
    input_amplitude_filename = 'final_amplitude_array.npy'
    output_amplitude_filename = 'final_amplitude_array_masked.npy'

    print(f"Generating masked amplitude array with size filter: {output_amplitude_filename}")
    current_directory = Path.cwd()
    subfolder_path = current_directory / directory_name

    # Load Image and Data Arrays
    image_path = subfolder_path / (strained_image_name + '_pol0.tiff')
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not open or find the image at {image_path}.")
        return

    x_array = np.load(subfolder_path / (strained_image_name + '_x_array.npy')).flatten()
    y_array = np.load(subfolder_path / (strained_image_name + '_y_array.npy')).flatten()
    amplitude_array = np.load(subfolder_path / input_amplitude_filename).flatten()

    # Perform Edge Detection
    row_start, row_end, column_start, column_end = bounding_box
    bbox_x = column_start
    bbox_y = row_start
    bbox_w = column_end - column_start
    bbox_h = row_end - row_start
    bbox_area = bbox_w * bbox_h  # Calculate the total area of the bounding box

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1)
    edges = cv2.Canny(blurred_image, 50, 200)

    # Contour Hierarchy Logic with Size Filter
    closed_edges = edges.copy()
    cv2.rectangle(closed_edges, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (255), 1)

    contours, hierarchy = cv2.findContours(closed_edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    feature_mask = np.zeros_like(gray_image, dtype=np.uint8)

    if hierarchy is not None:
        for i, contour in enumerate(contours):
            # Check if the contour has a parent (is a "hole")
            if hierarchy[0][i][3] != -1:
                # Calculate the contour's area
                contour_area = cv2.contourArea(contour)
                # Only fill contours that are smaller than 50% of the bounding box's area
                if contour_area < (bbox_area * 0.5):
                    cv2.drawContours(feature_mask, [contour], -1, (255), thickness=cv2.FILLED)

    # Apply Mask to Amplitude Array
    amplitude_array_modified = np.copy(amplitude_array)

    for i in range(len(x_array)):
        col = int(round(x_array[i]))
        row = int(round(y_array[i]))

        if 0 <= row < feature_mask.shape[0] and 0 <= col < feature_mask.shape[1]:
            if feature_mask[row, col] == 255:
                amplitude_array_modified[i] = np.nan

    # Save the New Masked Array
    output_path = subfolder_path / output_amplitude_filename
    np.save(output_path, amplitude_array_modified)
    print(f"Successfully saved masked array to {output_path}")


def contour_plots(directory_name, strained_image_name, bounding_box, gaussian_sigma=0, cmap='viridis', mask_edges=False,
                  log_scale=False, hide_colorbar=False, only_plot_amplitude=False, ax=None, show_plot=True):
    """Generates and saves contour plots for the final amplitude and phase data.

    This function visualizes the results of the vector subtraction by creating filled contour plots. It loads the final
    amplitude and phase arrays, along with a background image. The data is first interpolated onto a regular grid to
    ensure a smooth plot.

    Several pre-processing and customization options are available, including applying a Gaussian filter for smoothing,
    masking data at the edges to remove artifacts, and adjusting the plot's appearance with different colormaps or a
    logarithmic scale.

    Parameters
    ----------
    directory_name : str
        The subfolder containing the input `.npy` arrays and where the output plot will be saved.
    strained_image_name : str
        The base name used to load the background image and coordinate arrays.
    bounding_box : tuple
        The data region; required by the `mask_edges` option to define the masking area.
    gaussian_sigma : float, optional
        Standard deviation for the Gaussian smoothing filter applied to the data.
        If 0 (default), no smoothing is performed.
    cmap : str, optional
        The name of the Matplotlib colormap to use for the plots. Defaults to 'viridis'.
    mask_edges : bool, optional
        If True, applies a mask to exclude data near the edges of the bounding box, which can remove potential
        artifacts. Defaults to False.
    log_scale : bool, optional
        If True, the amplitude plot uses a logarithmic color scale, which can be useful for data with a large dynamic
        range. Defaults to False.
    hide_colorbar : bool, optional
        If True, the colorbar(s) are omitted from the plot. Defaults to False.
    only_plot_amplitude : bool, optional
        If False (default), creates a figure with two subplots for both amplitude and phase. If True, it generates a
        single plot of only the amplitude.
    ax : matplotlib.axes.Axes, optional
        An existing Matplotlib axis object on which to draw the amplitude plot. This parameter is **only** effective
        when `only_plot_amplitude` is set to True. Defaults to None.
    show_plot : bool, optional
        If True (default), displays the final plot in an interactive window.

    Returns
    -------
    None

    Side Effects
    ------------
    - Saves the generated contour plot as `contour_plot.png` in the specified directory.
    """

    print('Generating contour plots')
    current_directory = Path.cwd()
    subfolder_path = current_directory / directory_name

    # ... (The data loading and mask application logic remains the same) ...
    amplitude_array_orig = np.load(subfolder_path / 'final_amplitude_array.npy')
    phase_array_orig = np.load(subfolder_path / 'final_phase_array.npy')

    if gaussian_sigma > 0:
        amplitude_array = gaussian_filter(amplitude_array_orig, sigma=gaussian_sigma, mode='reflect')
        phase_array = gaussian_filter(phase_array_orig, sigma=gaussian_sigma, mode='wrap')
    else:
        amplitude_array = amplitude_array_orig
        phase_array = phase_array_orig

    if mask_edges:
        create_masked_amplitude_array(directory_name, strained_image_name, bounding_box)
        nan_mask_array_1d = np.load(subfolder_path / 'final_amplitude_array_masked.npy')

        try:
            target_shape = amplitude_array.shape
            nan_mask_array_2d = nan_mask_array_1d.reshape(target_shape)
        except ValueError:
            print(
                f"Error: Mismatch between data size ({amplitude_array.size}) and mask size ({nan_mask_array_1d.size}).")
            return

        amplitude_array[np.isnan(nan_mask_array_2d)] = np.nan
        phase_array[np.isnan(nan_mask_array_2d)] = np.nan

    pol0_img = Image.open(subfolder_path / (strained_image_name + '_pol0.tiff'))
    x_array = np.load(subfolder_path / (strained_image_name + '_x_array.npy'))
    y_array = np.load(subfolder_path / (strained_image_name + '_y_array.npy'))

    x_min, x_max = x_array.min(), x_array.max()
    y_min, y_max = y_array.min(), y_array.max()

    nx_new = 200
    ny_new = 200
    xi = np.linspace(x_min, x_max, nx_new)
    yi = np.linspace(y_min, y_max, ny_new)
    X_grid, Y_grid = np.meshgrid(xi, yi)

    points_orig = np.vstack((x_array.flatten(), y_array.flatten())).T

    amplitude_orig = amplitude_array.flatten()
    phase_orig = phase_array.flatten()
    amp_grid = griddata(points_orig, amplitude_orig, (X_grid, Y_grid), method='linear')
    phase_grid = griddata(points_orig, phase_orig, (X_grid, Y_grid), method='linear')

    # --- Define levels and normalization for the plot ---
    num_levels = 50
    norm_object = None
    levels_to_plot = num_levels

    if log_scale:
        # Find min/max of the data, ensuring they are positive for log scale
        vmin = np.nanmin(amp_grid[amp_grid > 0])
        vmax = np.nanmax(amp_grid)

        # Check if the data range is valid for a log scale
        if vmin > 0 and vmax > vmin:
            # Create logarithmically spaced levels from data min to max
            levels_to_plot = np.logspace(np.log10(vmin), np.log10(vmax), num_levels)
            norm_object = LogNorm(vmin=vmin, vmax=vmax)
        else:
            print(
                "Warning: Data not suitable for log scale (e.g., all zeros or negative values). Reverting to linear scale.")
            log_scale = False  # Revert to linear if data is invalid

    if only_plot_amplitude is False:
        if ax is not None:
            print('ax is being ignored because only_plot_amplitude is False.'
                  'To plot the amplitude on a specific axis, set only_plot_amplitude=True')
            handlefigure = True
        fig, ax = plt.subplots(1, 2, figsize=(18, 7))
        ax[0].imshow(pol0_img, cmap="gray")

        # Plot the contour using the selected levels and normalization
        cf = ax[0].contourf(X_grid, Y_grid, amp_grid, levels=levels_to_plot, cmap=cmap, norm=norm_object)
        if hide_colorbar is False:
            fig.colorbar(cf, ax=ax[0], orientation='vertical', label='Amplitude')

        ax[0].set_title("Amplitude")
        ax[0].set_aspect('equal', adjustable='box')
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        ax[1].imshow(pol0_img, cmap="gray")
        cf_phase = ax[1].contourf(X_grid, Y_grid, phase_grid, levels=num_levels, cmap=cmap)
        if hide_colorbar is False:
            fig.colorbar(cf_phase, ax=ax[1], orientation='vertical', label='Phase')
        ax[1].set_title("Phase")
        ax[1].set_aspect('equal', adjustable='box')
        ax[1].set_xticks([])
        ax[1].set_yticks([])

    else:
        if ax is not None:
            fig = plt.gcf()
            handlefigure = False
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            handlefigure = True
        ax.imshow(pol0_img, cmap="gray")

        # Plot the contour using the selected levels and normalization
        cf = ax.contourf(X_grid, Y_grid, amp_grid, levels=levels_to_plot, cmap=cmap, norm=norm_object)
        if hide_colorbar is False:
            fig.colorbar(cf, ax=ax, orientation='vertical', label='Amplitude')

        ax.set_title("Amplitude")
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(subfolder_path / 'contour_plot.png', bbox_inches='tight')

    if handlefigure is True:
        if show_plot:
            plt.show()
        else:
            plt.close(fig)


def quiver_plot(directory_name, strained_image_name, bounding_box, gaussian_sigma=0, mask_features=False, skip_rows=15,
                skip_cols=15, scale=1.0, cmap='viridis', head_width=3, head_length=4, show_plot=True):
    """
    Generates a quiver plot to visualize amplitude and phase data as vectors.

    Parameters
    ----------
    directory_name (str):
        The name of the directory containing the data files.
    strained_image_name (str):
        The base name of the strained image files.
    bounding_box (tuple):
        A tuple defining the ROI for masking.
    gaussian_sigma (int, optional):
        The sigma value for the Gaussian filter. If 0, no filter is applied. Defaults to 0.
    mask_features (bool, optional):
        If True, masks features like holes. Defaults to False.
    skip_rows (int, optional):
        The step size for vertical downsampling. Defaults to 15.
    skip_cols (int, optional):
        The step size for horizontal downsampling. Defaults to 15.
    scale (float, optional):
        Scaling factor for the arrow lengths. Defaults to 1.0.
    cmap (str, optional):
        The colormap for coloring arrows by magnitude. Defaults to 'viridis'.
    head_width (int, optional):
        The width of the arrow head in points. Defaults to 3.
    head_length (int, optional):
        The length of the arrow head in points. Defaults to 4.
    show_plot (bool, optional):
        If True, displays the plot. Defaults to True.

    Returns
    -------
    None
    """
    print('Generating quiver plot...')
    current_directory = Path.cwd()
    subfolder_path = current_directory / directory_name

    # Load Data
    amplitude_array = np.load(subfolder_path / 'final_amplitude_array.npy')
    phase_array = np.load(subfolder_path / 'final_phase_array.npy')

    # Gaussian filter
    if gaussian_sigma > 0:
        # Use 'reflect' mode for amplitude to handle edges
        amplitude_array = gaussian_filter(amplitude_array, sigma=gaussian_sigma, mode='reflect')
        # Use 'wrap' mode for phase to handle cyclical data (e.g., 2π wraps to 0)
        phase_array = gaussian_filter(phase_array, sigma=gaussian_sigma, mode='wrap')

    # Apply Mask
    if mask_features:
        create_masked_amplitude_array(directory_name, strained_image_name, bounding_box)
        nan_mask_array_1d = np.load(subfolder_path / 'final_amplitude_array_masked.npy')
        try:
            nan_mask_array_2d = nan_mask_array_1d.reshape(amplitude_array.shape)
            amplitude_array[np.isnan(nan_mask_array_2d)] = np.nan
            phase_array[np.isnan(nan_mask_array_2d)] = np.nan
        except ValueError:
            print(f"Error: Mismatch in mask and data size. Skipping masking.")

    pol0_img = Image.open(subfolder_path / (strained_image_name + '_pol0.tiff'))
    x_array = np.load(subfolder_path / (strained_image_name + '_x_array.npy'))
    y_array = np.load(subfolder_path / (strained_image_name + '_y_array.npy'))

    # Grid the data
    x_min, x_max = x_array.min(), x_array.max()
    y_min, y_max = y_array.min(), y_array.max()
    nx_new, ny_new = 200, 200
    xi = np.linspace(x_min, x_max, nx_new)
    yi = np.linspace(y_min, y_max, ny_new)
    X_grid, Y_grid = np.meshgrid(xi, yi)
    points_orig = np.vstack((x_array.flatten(), y_array.flatten())).T
    amp_grid = griddata(points_orig, amplitude_array.flatten(), (X_grid, Y_grid), method='linear')
    phase_grid = griddata(points_orig, phase_array.flatten(), (X_grid, Y_grid), method='linear')

    # Calculate Vector Components
    U = amp_grid * np.cos(phase_grid)
    V = amp_grid * np.sin(phase_grid)

    # Create the Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(pol0_img, cmap="gray")

    X_skip = X_grid[::skip_rows, ::skip_cols]
    Y_skip = Y_grid[::skip_rows, ::skip_cols]
    U_skip = U[::skip_rows, ::skip_cols]
    V_skip = V[::skip_rows, ::skip_cols]
    magnitude_skip = amp_grid[::skip_rows, ::skip_cols]

    q = ax.quiver(X_skip, Y_skip, U_skip, V_skip, magnitude_skip, cmap=cmap, scale=scale,
                  scale_units='xy', angles='xy', headwidth=head_width, headlength=head_length)

    ax.set_title("Amplitude and Phase Quiver Plot")
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(q, ax=ax, label='Magnitude (Amplitude)')

    # Save and display the plot
    plt.tight_layout()
    output_path = subfolder_path / 'quiver_plot_masked_filtered.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Successfully saved plot to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def correlation_plot(directory_name, bounding_box, strained_image_name, unstrained_image_name=None):
    """Analyzes and plots the linear correlation between pixel intensities.

    This function investigates the relationship between pairs of polarization images (e.g., 0° vs 90°) for both strained
    and optional unstrained datasets.

    For each pixel within a specified bounding box, it normalizes the intensity by dividing it by that pixel's average
    intensity across all four polarization angles (I / I_ave). It then generates a 2x3 grid of scatter plots, where each
    subplot compares a different pair of the normalized polarization intensities. A line of best fit (y = mx + c) is
    calculated and overlaid on the data, with the R-squared value displayed in the legend to quantify the correlation.

    Parameters
    ----------
    directory_name : str
        The subfolder containing the input polarization image files.
    bounding_box : tuple
        A tuple of four integers `(row_start, row_end, col_start, col_end)` that defines the rectangular region of
        interest for the analysis.
    strained_image_name : str
        The base name for the set of strained polarization images.
    unstrained_image_name : str, optional
        The base name for the unstrained image set. If None (default), this dataset is skipped, and only the strained
        data is analyzed.

    Returns
    -------
    None

    Side Effects
    ------------
    - Displays a Matplotlib figure containing six subplots. The plot is interactive and remains open until closed by the
    user.

    """

    print('Fitting lines of best fit to polarisation image pixels (I / I_ave) with y=mx+c')
    current_directory = Path.cwd()
    subfolder_path = current_directory / directory_name
    row_start, row_end, column_start, column_end = bounding_box

    # Create a 2x3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    pointcolors = ['darkorange', 'steelblue']
    linecolors = ['red', 'blue']
    names = ['unstrained', 'strained']

    for image_idx, image_name in enumerate([unstrained_image_name, strained_image_name]):

        if image_name is not None:

            # Define image paths
            base_paths = {
                'pol0': subfolder_path / (image_name + '_pol0.tiff'),
                'pol45': subfolder_path / (image_name + '_pol45.tiff'),
                'pol90': subfolder_path / (image_name + '_pol90.tiff'),
                'pol135': subfolder_path / (image_name + '_pol135.tiff')
            }

            # Load images and convert to NumPy arrays
            img_arrays_raw = {}
            for key, path in base_paths.items():
                try:
                    img = Image.open(path)
                    img_arrays_raw[key] = np.array(img)
                except FileNotFoundError:
                    print(f"Error: Image file not found at {path}")
                    return
                except Exception as e:
                    print(f"Error loading image {path}: {e}")
                    return

            if len(img_arrays_raw) != 4:
                print("Error: Not all four polarization images were loaded.")
                return

            # Extract ROIs
            rois_raw = {}
            for key, arr in img_arrays_raw.items():
                roi = arr[row_start:row_end, column_start:column_end]
                rois_raw[key] = roi.astype(np.float64)  # Convert to float for calculations

            # Calculate the pixel-wise average intensity across the 4 ROIs
            stacked_rois = np.stack([rois_raw['pol0'], rois_raw['pol45'], rois_raw['pol90'], rois_raw['pol135']],
                                    axis=0)
            roi_avg_intensity = np.mean(stacked_rois, axis=0)

            # Subtract the average intensity ROI from each polarization ROI
            rois_mod = {}
            for key in rois_raw.keys():
                rois_mod[key] = rois_raw[key] / roi_avg_intensity

            # Flatten the modified ROIs
            rois_flat = {}
            for key, arr in rois_mod.items():
                rois_flat[key] = arr.flatten()

            all_mod_intensities = np.concatenate(list(rois_flat.values()))
            if all_mod_intensities.size == 0:
                print("Error: No data points found in ROIs after processing.")
                return

            plot_combinations = [
                (rois_flat['pol0'], rois_flat['pol90'], '0', '90'),
                (rois_flat['pol45'], rois_flat['pol135'], '45', '135'),
                (rois_flat['pol0'], rois_flat['pol45'], '0', '45'),
                (rois_flat['pol0'], rois_flat['pol135'], '0', '135'),
                (rois_flat['pol45'], rois_flat['pol90'], '45', '90'),
                (rois_flat['pol90'], rois_flat['pol135'], '90', '135')
            ]

            for i, combo in enumerate(plot_combinations):
                ax_current = axes[i]
                x_intensities_mod = combo[0]
                y_intensities_mod = combo[1]
                label_x = combo[2]
                label_y = combo[3]

                if x_intensities_mod.size < 2 or y_intensities_mod.size < 2:
                    print(
                        f"Warning: Not enough data points for regression in {label_x}° vs {label_y}° (found {x_intensities_mod.size} points). Skipping fit.")
                    ax_current.scatter(x_intensities_mod, y_intensities_mod, color=pointcolors[image_idx], s=5,
                                       alpha=0.2, label='Pixel (I - $I_{ave}$) ' + names[image_idx])
                    fit_label = "Fit (Not enough data)"
                    slope, intercept, r_squared_val = np.nan, np.nan, np.nan
                else:
                    # Scatter plot for the modified pixel intensities
                    ax_current.scatter(x_intensities_mod, y_intensities_mod, color=pointcolors[image_idx], s=5,
                                       alpha=0.2, label='Pixel $(I / I_{{ave}})$ ' + names[image_idx])

                    # Perform linear regression using scipy.stats.linregress
                    try:
                        slope, intercept, r_value, p_value, std_err = linregress(x_intensities_mod, y_intensities_mod)
                        r_squared_val = r_value ** 2
                        fit_label = f'y={slope:.2f}x + {intercept:.2f}\nR²={r_squared_val:.2f} '
                    except ValueError as e:
                        print(f"ValueError during linregress for {label_x} vs {label_y}: {e}")
                        slope, intercept, r_squared_val = np.nan, np.nan, np.nan
                        fit_label = 'Fit (Error)'

                # Create line points for plotting
                min_x_mod = np.min(x_intensities_mod)
                max_x_mod = np.max(x_intensities_mod)
                x_line_points = np.array([min_x_mod, max_x_mod])

                if not np.isnan(slope) and not np.isnan(intercept):  # Only plot if fit was successful
                    y_line_points = slope * x_line_points + intercept
                    ax_current.plot(x_line_points, y_line_points, color=linecolors[image_idx], linewidth=1.5,
                                    label=fit_label + names[image_idx])
                elif fit_label == "Fit (Not enough data)" and x_intensities_mod.size > 0:  # Still add legend if only scatter
                    ax_current.legend(fontsize=9)

                ax_current.set_title(f'{label_x}° vs {label_y}° $\\left(\\frac{{I}}{{I_{{ave}}}}\\right)$', fontsize=12,
                                     weight='bold')
                ax_current.set_xlabel(f'$\\frac{{I({label_x}°)}}{{I_{{ave}}}}$')
                ax_current.set_ylabel(f'$\\frac{{I({label_y}°)}}{{I_{{ave}}}}$')
                if fit_label != "Fit (Not enough data)" or x_intensities_mod.size == 0:  # avoid double legend
                    ax_current.legend(fontsize=9)  # Adjusted fontsize
                ax_current.grid(True, linestyle=':', alpha=0.7)

            xlim = ax_current.get_xlim()
            ylim = ax_current.get_ylim()
            ax_current.axhline(1, color='black', linewidth=0.5, linestyle='--')
            ax_current.axvline(1, color='black', linewidth=0.5, linestyle='--')
            ax_current.set_xlim(xlim)
            ax_current.set_ylim(ylim)

    fig.suptitle('Pixel Intensity Correlations & Linear Fits $\\left(\\frac{{I}}{{I_{ave}}}, y=mx+c\\right)$',
                 fontsize=14, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def vector_plot(directory_name, unstrained_image_name, strained_image_name, plot_type='cartesian'):
    """Creates a scatter plot to visualize vector data distribution.

    This function loads the amplitude and phase data for the unstrained, strained, and final (difference) vector fields
    from their respective NumPy files. It then generates a 2D scatter plot showing the distribution of these vectors
    from the origin.

    The plot can be rendered in either Cartesian (x, y) or polar(amplitude vs. phase) coordinates, controlled by the
    `plot_type` parameter. Each dataset is plotted in a distinct color for comparison.

    Parameters
    ----------
    directory_name : str
        The subfolder containing the input `.npy` array files.
    unstrained_image_name : str
        The base name used to load the unstrained amplitude and phase arrays.
    strained_image_name : str
        The base name used to load the strained amplitude and phase arrays.
    plot_type : str, optional
        The coordinate system for the plot. Accepts 'cartesian' (default)
        to plot x vs. y, or 'polar' to plot amplitude vs. phase directly.

    Returns
    -------
    None

    Side Effects
    ------------
    - Displays an interactive Matplotlib scatter plot.
    """

    current_directory = Path.cwd()
    subfolder_path = current_directory / directory_name
    unstrained_amplitude_array = np.load(subfolder_path / (unstrained_image_name + '_amplitude_array.npy'))
    unstrained_phase_array = np.load(subfolder_path / (unstrained_image_name + '_phase_array.npy'))
    strained_amplitude_array = np.load(subfolder_path / (strained_image_name + '_amplitude_array.npy'))
    strained_phase_array = np.load(subfolder_path / (strained_image_name + '_phase_array.npy'))
    final_amplitude_array = np.load(subfolder_path / 'final_amplitude_array.npy')
    final_phase_array = np.load(subfolder_path / 'final_phase_array.npy')

    if plot_type == 'cartesian':
        # Convert polar coordinates (amplitude, phase) to Cartesian coordinates (x, y)
        x_unstrained = unstrained_amplitude_array * np.cos(unstrained_phase_array)
        y_unstrained = unstrained_amplitude_array * np.sin(unstrained_phase_array)
        x_strained = strained_amplitude_array * np.cos(strained_phase_array)
        y_strained = strained_amplitude_array * np.sin(strained_phase_array)
        x_final = final_amplitude_array * np.cos(final_phase_array)
        y_final = final_amplitude_array * np.sin(final_phase_array)
        plt.scatter(x_final, y_final, alpha=0.2, marker='.', color='green', label='final')
        plt.scatter(x_unstrained, y_unstrained, alpha=0.2, marker='.', color='red', label='unstrained')
        plt.scatter(x_strained, y_strained, alpha=0.2, marker='.', color='steelblue', label='strained')

    elif plot_type == 'polar':
        plt.subplots(subplot_kw={'projection': 'polar'})
        plt.scatter(final_phase_array, final_amplitude_array, alpha=0.2, marker='.', color='green', label='final')
        plt.scatter(unstrained_phase_array, unstrained_amplitude_array, alpha=0.2, marker='.', color='red',
                    label='unstrained')
        plt.scatter(strained_phase_array, strained_amplitude_array, alpha=0.2, marker='.', color='steelblue',
                    label='strained')

    plt.title("Vectors from Amplitudes and Phases")
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.show()


def PDF_plot(directory_name, bounding_box, image_name, combined=False):
    """Plots the Probability Density Function (PDF) of pixel intensities.

    This function analyzes the statistical distribution of pixel intensity values within a specified region of interest
    (ROI) for the four polarization images (0°, 45°, 90°, 135°).

    It generates normalized histograms to represent the empirical PDF of the intensities. It then fits several
    theoretical probability distributions (e.g., Normal, Weibull Mixture, Gamma) to the data to model its behavior. The
    resulting plots show both the raw histogram and the fitted distribution curves.

    Parameters
    ----------
    directory_name : str
        The subfolder containing the input polarization image files.
    bounding_box : tuple
        A tuple of four integers `(row_start, row_end, col_start, col_end)` that defines the rectangular ROI for the
        analysis.
    image_name : str
        The base name for the set of polarization images to be analyzed.
    combined : bool, optional
        Controls the plot layout.
        If False (default), a 2x2 grid of subplots is created, one for each polarization.
        If True, all four distributions are overlaid on a single plot for direct comparison.

    Returns
    -------
    None

    Side Effects
    ------------
    Displays an interactive Matplotlib figure showing the PDF plots.
    """

    print('Plotting PDF of polarisation image pixels')
    current_directory = Path.cwd()
    subfolder_path = current_directory / directory_name
    row_start, row_end, column_start, column_end = bounding_box

    # Define image paths
    base_paths = {
        'pol0': subfolder_path / (image_name + '_pol0.tiff'),
        'pol45': subfolder_path / (image_name + '_pol45.tiff'),
        'pol90': subfolder_path / (image_name + '_pol90.tiff'),
        'pol135': subfolder_path / (image_name + '_pol135.tiff')
    }

    # Load images and convert to NumPy arrays
    img_arrays = {}
    for key, path in base_paths.items():
        img = Image.open(path)
        img_arrays[key] = np.array(img)

    # Extract ROIs and flatten
    rois_flat = {}
    for key, arr in img_arrays.items():
        roi = arr[row_start:row_end, column_start:column_end]
        rois_flat[key] = roi.flatten()

    plot_combinations = [
        (rois_flat['pol0'], '0'),
        (rois_flat['pol45'], '45'),
        (rois_flat['pol90'], '90'),
        (rois_flat['pol135'], '135')
    ]

    if combined == True:
        fig, axes = plt.subplots(1, 1, figsize=(12, 8))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()  # Flatten the 2D array of axes for easier iteration
        fig.suptitle('PDF of Pixel Intensities & fitted distributions', fontsize=14)

    colors = ['red', 'blue', 'green', 'orange']
    for i, combo in enumerate(plot_combinations):
        intensities = combo[0]
        label = combo[1]
        if combined == True:
            histogram(intensities, density=True, bins=20, color=colors[i], alpha=0.5, label=label)
            normal_fit = Fit_Normal_2P(intensities, show_probability_plot=False, print_results=False)
            normal_fit.distribution.PDF(label='Normal 2P for pol ' + label)
            plt.title('PDF for all polarisations', weight='bold')
        else:
            plt.sca(axes[i])  # Get the current subplot axis
            histogram(intensities, density=True, bins=20, color=colors[i], alpha=0.5)
            # weibull_fit = Fit_Weibull_3P(intensities,show_probability_plot=False, print_results=False)
            # normal_fit = Fit_Normal_2P(intensities,show_probability_plot=False, print_results=False)
            mixture_fit = Fit_Weibull_Mixture(intensities, show_probability_plot=False, print_results=False)
            gamma_fit = Fit_Gamma_3P(intensities, show_probability_plot=False, print_results=False)
            # weibull_fit.distribution.PDF(label='Weibull 3P')
            # normal_fit.distribution.PDF(label='Normal 2P')
            mixture_fit.distribution.PDF(label='Weibull Mixture')
            gamma_fit.distribution.PDF(label='Gamma 3P')
            plt.title('PDF for Polarisation ' + label + '°', weight='bold')
        plt.xlim(min(intensities), max(intensities))
        plt.xlabel('Pixel Intensities')
        plt.legend(loc='upper left')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def histogram_plots(directory_name, image_name, ln=False):
    """Generates histograms for the final amplitude and phase data.

    This function loads the final calculated amplitude and phase arrays and creates histograms to visualize their
    statistical distributions. It can display the distribution of the raw data, the natural logarithm (ln) of the data,
    or both side-by-side.

    Before calculating logarithms, it applies minor corrections to the data to handle zeros in amplitude and negative
    values in phase, ensuring numerical stability.

    Parameters
    ----------
    directory_name : str
        The subfolder containing the input `.npy` array files.
    image_name : str
        The base name used to load the coordinate arrays.
    ln : bool or str, optional
        Controls the plot content.
        False (default): Plots histograms of the original amplitude and phase.
        True: Plots histograms of the natural log of amplitude and phase.
        'both': Creates a 2x2 grid showing all four plots.

    Returns
    -------
    None

    Side Effects
    ------------
    Displays an interactive Matplotlib figure with the requested histograms.
    """

    print('Generating histogram plots')
    current_directory = Path.cwd()
    # Create the Path object for the new subfolder
    subfolder_path = current_directory / directory_name

    x_array = np.load(subfolder_path / (image_name + '_x_array.npy'))
    y_array = np.load(subfolder_path / (image_name + '_y_array.npy'))
    amplitude_array0 = np.load(subfolder_path / 'final_amplitude_array.npy')
    phase_array0 = np.load(subfolder_path / 'final_phase_array.npy')

    amplitude_array = np.asarray(amplitude_array0).flatten()
    phase_array = np.asarray(phase_array0).flatten()

    amplitude_array2 = np.copy(amplitude_array)
    phase_array2 = np.copy(phase_array)

    amplitude_array2[amplitude_array2 == 0] += 1e-6  # correction reqired for plotting ln(0)
    phase_array2[phase_array2 <= 0] += 2 * np.pi  # add 2 pi to any negative phase values to make them all positive
    ln_amplitude_array = np.log(amplitude_array2)
    ln_phase_array = np.log(phase_array2)

    bins = 100
    if ln == False:
        plt.subplot(121)
        histogram(amplitude_array, color='red', alpha=0.7, label='original', bins=bins)
        plt.title('Amplitude')
        plt.subplot(122)
        histogram(phase_array, color='red', alpha=0.7, label='original', bins=bins)
        plt.title('Phase (degrees)')
        plt.suptitle('Histogram of Amplitude and Phase')
    if ln == True:
        plt.subplot(121)
        histogram(ln_amplitude_array, color='blue', alpha=0.7, label='ln', bins=bins)
        plt.title('ln(Amplitude)')
        plt.subplot(122)
        histogram(ln_phase_array, color='blue', alpha=0.7, label='ln', bins=bins)
        plt.title('ln(Phase)')
        plt.suptitle('Histogram of ln(Amplitude) and ln(Phase)')
    if ln == 'both':
        plt.subplot(221)
        histogram(amplitude_array, color='red', alpha=0.7, label='original', bins=bins)
        plt.title('Amplitude')
        plt.subplot(222)
        histogram(phase_array, color='red', alpha=0.7, label='original', bins=bins)
        plt.title('Phase (degrees)')
        plt.subplot(223)
        histogram(ln_amplitude_array, color='blue', alpha=0.7, label='ln', bins=bins)
        plt.title('ln(Amplitude)')
        plt.subplot(224)
        histogram(ln_phase_array, color='blue', alpha=0.7, label='ln', bins=bins)
        plt.title('ln(Phase)')
        plt.suptitle('Histogram of Amplitude, ln(Amplitude), Phase, and ln(Phase)')
        plt.gcf().set_size_inches((10, 7))
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for suptitle
    plt.show()


def draw_bounding_box(directory_name, bounding_box, unstrained_image_name, strained_image_name):
    """Visualizes the specified bounding box on the unstrained and strained images.

    This function loads the 0° polarization images for both the unstrained and strained datasets. It displays them
    side-by-side in a Matplotlib window and draws a red rectangular overlay on each image, corresponding to the provided
    bounding box coordinates.

    This provides a clear visual confirmation of the selected region of interest before further processing.

    Parameters
    ----------
    directory_name : str
        The subfolder containing the input image files.
    bounding_box : tuple
        A tuple of four integers `(row_start, row_end, col_start, col_end)` that defines the rectangle to be drawn.
    unstrained_image_name : str
        The base name for the unstrained `_pol0.tiff` image.
    strained_image_name : str
        The base name for the strained `_pol0.tiff` image.

    Returns
    -------
    None

    Side Effects
    ------------
    Displays an interactive Matplotlib figure showing the two images with the bounding box overlay.
    """

    print('Drawing bounding box')
    current_directory = Path.cwd()
    # Create the Path object for the new subfolder
    subfolder_path = current_directory / directory_name

    row_start, row_end, column_start, column_end = bounding_box
    # Create figure and axes
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    for idx, image_name in enumerate([unstrained_image_name, strained_image_name]):
        pol0_img_path = subfolder_path / (image_name + '_pol0.tiff')
        pol0_img_pil = Image.open(pol0_img_path)  # Keep PIL image for imshow

        ax[idx].imshow(pol0_img_pil, cmap="gray")  # Display the image

        # Create a Rectangle patch
        rect_width = column_end - column_start
        rect_height = row_end - row_start
        rectangle = patches.Rectangle(
            (column_start, row_start),  # (x, y) of the bottom-left corner of the rectangle
            rect_width,  # width of the rectangle
            rect_height,  # height of the rectangle
            linewidth=2,  # thickness of the rectangle border
            edgecolor='red',  # color of the border
            facecolor='none'  # no fill color
        )

        ax[idx].add_patch(rectangle)  # Add the rectangle to the Axes
        if idx == 0:
            ax[idx].set_title('Unstrained image')
        elif idx == 1:
            ax[idx].set_title('Strained image')
        ax[idx].axis('off')
    plt.suptitle(f'Bounding Box: Rows [{row_start}-{row_end - 1}], Columns [{column_start}-{column_end - 1}]')
    plt.show()


def intensity_violins(directory_name, bounding_box, image_name):
    """Creates a violin plot to compare normalized pixel intensity distributions.

    This function visualizes and compares the statistical distribution of pixel intensities from the four polarization
    images. For each pixel within a specified bounding box, it normalizes the intensity by dividing it by that pixel's
    average intensity across all four polarization angles (I / I_ave).

    It then generates a single figure containing four violin plots, one for each polarization angle. This allows for a
    direct visual comparison of the shape, median, and spread of the normalized intensity distributions.

    Parameters
    ----------
    directory_name : str
        The subfolder containing the input polarization image files.
    bounding_box : tuple
        A tuple of four integers `(row_start, row_end, col_start, col_end)` that defines the rectangular region of
        interest for the analysis.
    image_name : str
        The base name for the set of polarization images to be analyzed.

    Returns
    -------
    None

    Side Effects
    ------------
    Displays an interactive Matplotlib figure containing the violin plots.
    """

    print('Plotting intensity violins at each polarisation')
    current_directory = Path.cwd()
    subfolder_path = current_directory / directory_name
    row_start, row_end, column_start, column_end = bounding_box

    # Define image paths
    base_paths = {
        'pol0': subfolder_path / (image_name + '_pol0.tiff'),
        'pol45': subfolder_path / (image_name + '_pol45.tiff'),
        'pol90': subfolder_path / (image_name + '_pol90.tiff'),
        'pol135': subfolder_path / (image_name + '_pol135.tiff')
    }

    # Load images and convert to NumPy arrays
    img_arrays_raw = {}
    for key, path in base_paths.items():
        try:
            img = Image.open(path)
            img_arrays_raw[key] = np.array(img)
        except FileNotFoundError:
            print(f"Error: Image file not found at {path}")
            return
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return

    if len(img_arrays_raw) != 4:
        print("Error: Not all four polarization images were loaded.")
        return

    # Extract ROIs
    rois_raw = {}
    for key, arr in img_arrays_raw.items():
        roi = arr[row_start:row_end, column_start:column_end]
        rois_raw[key] = roi.astype(np.float64)  # Convert to float for calculations

    # Calculate the pixel-wise average intensity across the 4 ROIs
    stacked_rois = np.stack([rois_raw['pol0'], rois_raw['pol45'], rois_raw['pol90'], rois_raw['pol135']], axis=0)
    roi_avg_intensity = np.mean(stacked_rois, axis=0)

    # Subtract the average intensity ROI from each polarization ROI
    rois_mod = {}
    for key in rois_raw.keys():
        rois_mod[key] = rois_raw[key] / roi_avg_intensity

    # Flatten the modified ROIs
    rois_flat = {}
    for key, arr in rois_mod.items():
        rois_flat[key] = arr.flatten()

    # Data to plot - a list of your arrays
    pol0_intensities = rois_flat['pol0']
    pol45_intensities = rois_flat['pol45']
    pol90_intensities = rois_flat['pol90']
    pol135_intensities = rois_flat['pol135']
    data_to_plot = [pol0_intensities, pol45_intensities, pol90_intensities, pol135_intensities]

    # Create a figure and an axes
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figsize as needed

    # Create the violin plot
    # showmedians=True will draw a line at the median of each distribution
    # quantiles can be used to show specific quantiles, e.g., [[0.25, 0.75], [0.25, 0.75]...] for each violin
    violin_parts = ax.violinplot(data_to_plot, showmedians=True, showextrema=True)

    # Customize the plot
    ax.set_xticks(np.arange(1, len(data_to_plot) + 1))  # Set x-tick positions
    ax.set_xticklabels(['0°', '45°', '90°', '135°'])  # Set x-tick labels
    ax.set_xlabel('Polarization Angle', fontsize=12)
    ax.set_ylabel('Pixel Intensity', fontsize=12)
    ax.set_title('Distribution of Pixel Intensities by Polarization for ' + directory_name, fontsize=14, weight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.7)  # Add a light grid for the y-axis

    # Customizing colors of the violins (optional)
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'plum']
    for pc, color in zip(violin_parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.8)

    # Customizing medians and extrema lines (optional)
    if 'cmedians' in violin_parts:
        violin_parts['cmedians'].set_edgecolor('red')
        violin_parts['cmedians'].set_linewidth(2)
    if 'cmins' in violin_parts:
        violin_parts['cmins'].set_edgecolor('gray')
    if 'cmaxes' in violin_parts:
        violin_parts['cmaxes'].set_edgecolor('gray')
    if 'cbars' in violin_parts:  # This is the bar connecting min/max through the kde
        violin_parts['cbars'].set_edgecolor('gray')

    plt.tight_layout()
    plt.show()


def plot_intensities(directory_sup_name, angles, bounding_box):
    """Plots average ROI intensity across a series of experimental angles.

    This function analyzes image data from multiple experimental runs, with each run corresponding to a different angle
    provided in the `angles` list.

    For each experimental angle, it loads the four polarization images (0°, 45°, 90°, 135°), calculates the mean
    intensity within the specified bounding box for each, and stores these values. It then generates a composite plot:
    1.  The top part is a line graph showing how the average intensity for each of the four polarizations changes as a
        function of the experimental angle.
    2.  The bottom part is a row of the 0° polarization images, one from each experiment, with the bounding box drawn on
        them as a visual reference.

    Parameters
    ----------
    directory_sup_name : str
        The prefix for the subfolder names. Each subfolder is expected to be named `f"{directory_sup_name}{angle}"`.
    angles : list of str
        A list of strings representing the different experimental angles to be processed (e.g., ['0', '15', '30']).
    bounding_box : tuple
        A tuple of four integers `(row_start, row_end, col_start, col_end)` that defines the rectangular ROI for
        calculating average intensity.

    Returns
    -------
    None

    Side Effects
    ------------
    Displays an interactive Matplotlib figure.
    """

    print('Plotting intensities')
    int_angles = [int(s) for s in angles]

    current_directory = Path.cwd()
    # Create the Path object for the new subfolder

    row_start, row_end, column_start, column_end = bounding_box

    avg_intensity_0 = np.zeros_like(angles, dtype=float)
    avg_intensity_45 = np.zeros_like(angles, dtype=float)
    avg_intensity_90 = np.zeros_like(angles, dtype=float)
    avg_intensity_135 = np.zeros_like(angles, dtype=float)

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(nrows=5, ncols=len(angles), wspace=0, hspace=0)

    for idx, angle in enumerate(angles):
        subfolder_path = current_directory / (directory_sup_name + angle)

        # Define image paths
        base_paths = {
            'pol0': subfolder_path / ('STRAINED' + angle + '_pol0.tiff'),
            'pol45': subfolder_path / ('STRAINED' + angle + '_pol45.tiff'),
            'pol90': subfolder_path / ('STRAINED' + angle + '_pol90.tiff'),
            'pol135': subfolder_path / ('STRAINED' + angle + '_pol135.tiff')
        }

        # Load images and convert to NumPy arrays
        img_arrays_raw = {}
        for key, path in base_paths.items():
            try:
                img = Image.open(path)
                img_arrays_raw[key] = np.array(img)
            except FileNotFoundError:
                print(f"Error: Image file not found at {path}")
                return
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                return

        if len(img_arrays_raw) != 4:
            print("Error: Not all four polarization images were loaded.")
            return

        # Extract ROIs
        rois_raw = {}
        for key, arr in img_arrays_raw.items():
            roi = arr[row_start:row_end, column_start:column_end]
            rois_raw[key] = roi.astype(np.float64)  # Convert to float for calculations

        avg_intensity_0[idx] = np.mean(rois_raw['pol0'])
        avg_intensity_45[idx] = np.mean(rois_raw['pol45'])
        avg_intensity_90[idx] = np.mean(rois_raw['pol90'])
        avg_intensity_135[idx] = np.mean(rois_raw['pol135'])

        ax_bottom = fig.add_subplot(gs[4, idx])
        ax_bottom.set_title(angle + '°')
        ax_bottom.imshow(img_arrays_raw['pol0'], cmap="gray")  # Display the image
        ax_bottom.set_xticks([])
        ax_bottom.set_yticks([])

        # Create a Rectangle patch
        rect_width = column_end - column_start
        rect_height = row_end - row_start
        rectangle = patches.Rectangle(
            (column_start, row_start),  # (x, y) of the bottom-left corner of the rectangle
            rect_width,  # width of the rectangle
            rect_height,  # height of the rectangle
            linewidth=1,  # thickness of the rectangle border
            edgecolor='red',  # color of the border
            facecolor='none'  # no fill color
        )
        ax_bottom.add_patch(rectangle)  # Add the rectangle to the Axes

    # Add the top subplot that spans all columns
    ax_top = fig.add_subplot(gs[0:3, :])
    ax_top.set_title('Intensity Plot')

    ax_top.plot(int_angles, avg_intensity_0, color='steelblue', label='0°')
    ax_top.plot(int_angles, avg_intensity_45, color='green', label='45°')
    ax_top.plot(int_angles, avg_intensity_90, color='darkorange', label='90°')
    ax_top.plot(int_angles, avg_intensity_135, color='red', label='135°')

    # Customize the plot
    ax_top.set_xlabel('Angle of PEC relative to camera', fontsize=12)
    ax_top.set_xticks(int_angles)
    ax_top.set_ylabel('Pixel Intensity', fontsize=12)
    ax_top.set_title('Average pixel intensities by polarization across a range of PEC angles', fontsize=14,
                     weight='bold')
    ax_top.grid(axis='y', linestyle='--', alpha=0.7)  # Add a light grid for the y-axis
    ax_top.legend()
    plt.show()
