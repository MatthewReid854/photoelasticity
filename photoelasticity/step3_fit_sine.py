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


import multiprocessing
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.optimize import OptimizeWarning, curve_fit
from tqdm import tqdm
from matplotlib.patches import Patch


def photoelastic_sine_function(alpha, F, G):
    """
    Sine function in the form described in the academic papers that use PVST.
    y = 1 + F * sin(2*alpha - 2*G)
    alpha: Analyser angle in radians.
    F: Optical Strain Response (OSR) - Amplitude.
    G: Phase of the OSR.
    """
    return 1 + F * np.sin(2 * alpha - 2 * G)


def process_pixel(task_data):
    """
    Performs a sine curve fit for a single pixel's data using the photoelastic since function.

    Args:
        task_data (tuple): Contains (i, j, p0, p45, p90, p135) pixel data.

    Returns:
        tuple: Contains (i, j, F_fit, G_fit, status_code, mape).
               status_code is 1 for success, -1 for failure.
               mape is the Mean Absolute Percentage Error.
    """
    i, j, p0, p45, p90, p135 = task_data

    # Angles must be in radians for the sine function
    x_data_rad = np.array([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])
    y_data = np.array([p0, p45, p90, p135])

    # Normalize the intensity data by the average intensity
    I_avg = np.mean(y_data)
    if I_avg == 0:  # Avoid division by zero
        return i, j, 0, 0, -1, np.nan

    y_data_normalized = y_data / I_avg

    # Provide initial guesses and bounds for F and G
    F_guess = np.std(y_data_normalized) * np.sqrt(2)
    initial_guesses = [F_guess, 0]  # Guess F=stdev, G=0
    # bounds = ([0, -np.pi], [1.5, np.pi])  # Bounds: F is [0, 1.5], G is [-pi, pi]
    bounds = ([0, -np.pi], [1, np.pi])  # Bounds: F is [0, 1.5], G is [-pi, pi]

    try:
        # Fit the photoelastic sine function to the normalized data
        params, _ = curve_fit(
            photoelastic_sine_function,
            x_data_rad,
            y_data_normalized,
            p0=initial_guesses,
            bounds=bounds,
            method="trf"
        )
        F_fit, G_fit = params
        status = 1

        # Calculate Mean Absolute Percentage Error (MAPE)
        y_fit = photoelastic_sine_function(x_data_rad, F_fit, G_fit)
        # Compare with the *normalized* true values
        non_zero_mask = y_data_normalized != 0
        if not np.any(non_zero_mask):
            mape = 0.0
        else:
            mape = np.mean(np.abs(
                (y_data_normalized[non_zero_mask] - y_fit[non_zero_mask]) / y_data_normalized[non_zero_mask])) * 100

    except (RuntimeError, ValueError):
        F_fit, G_fit = 0, 0
        status = -1  # Fail
        mape = np.nan

    return i, j, F_fit, G_fit, status, mape


def fit_sine_functions(directory_name, image_name, bounding_box, show_plot=True):
    """Fits a sinusoidal function to each pixel using four polarization images.

    This function analyzes a specified rectangular region within a set of four polarization images (0°, 45°, 90°, 135°).
    It treats the four intensity values at each pixel location as points along a sine curve.

    To handle the computationally intensive task of curve fitting for every pixel, it uses a multiprocessing pool to
    parallelize the workload across all available CPU cores. For each pixel, it calculates the amplitude (F) and phase
    (G) of the best-fit sine curve.

    The resulting data arrays are saved to disk as NumPy files. Additionally, two summary plots are generated and saved
    to visually assess the quality and accuracy of the fits across the region of interest.

    Parameters
    ----------
    directory_name : str
        The subfolder containing the input images and where output files will be saved.
    image_name : str
        The base name of the image files (e.g., 'unstrained_corrected').
    bounding_box : tuple
        A tuple of four integers `(row_start, row_end, col_start, col_end)`
        defining the rectangular region of interest to be analyzed.
    show_plot : bool, optional
        If True (default), displays the summary plots in interactive windows after saving them.

    Returns
    -------
    None

    Side Effects
    ------------
    - Saves four NumPy array files (`.npy`):
        - `{image_name}_amplitude_array.npy`: The calculated amplitude (F).
        - `{image_name}_phase_array.npy`: The calculated phase (G).
        - `{image_name}_x_array.npy`: X-coordinates for the analyzed region.
        - `{image_name}_y_array.npy`: Y-coordinates for the analyzed region.
    - Saves two PNG image files (`.png`):
        - `{image_name}_sine_fit_summary_mp.png`: An overlay showing the
          success (green) or failure (red) of the fit for each pixel.
        - `{image_name}_sine_fit_error_mp.png`: A contour plot overlay
          showing the Mean Absolute Percentage Error (MAPE) for each fit.

    """

    # Setup and load files
    current_directory = Path.cwd()
    subfolder_path = current_directory / directory_name
    subfolder_path.mkdir(exist_ok=True)

    warnings.simplefilter("ignore", OptimizeWarning)
    np.set_printoptions(precision=3, suppress=True)

    row_start, row_end, column_start, column_end = bounding_box

    try:
        pol0_img = Image.open(subfolder_path / f'{image_name}_pol0.tiff')
        pol45_img = Image.open(subfolder_path / f'{image_name}_pol45.tiff')
        pol90_img = Image.open(subfolder_path / f'{image_name}_pol90.tiff')
        pol135_img = Image.open(subfolder_path / f'{image_name}_pol135.tiff')
    except FileNotFoundError as e:
        print(f"Error: Could not find image file. {e}")
        return

    pol0_arr = np.array(pol0_img)
    pol45_arr = np.array(pol45_img)
    pol90_arr = np.array(pol90_img)
    pol135_arr = np.array(pol135_img)

    # Prepare data for parallel processing
    tasks = []
    for i in range(row_start, row_end):
        for j in range(column_start, column_end):
            task_data = (i, j, pol0_arr[i, j], pol45_arr[i, j], pol90_arr[i, j], pol135_arr[i, j])
            tasks.append(task_data)

    # Run tasks in parallel
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        print(f"Fitting sine functions with multiprocessing on {multiprocessing.cpu_count()} CPU cores")
        results = list(tqdm(pool.imap(process_pixel, tasks), total=len(tasks), desc="Fitting Curves"))

    # Aggregate the results
    output_rows = row_end - row_start
    output_cols = column_end - column_start
    F_array = np.zeros((output_rows, output_cols))  # OSR Amplitude F
    G_array = np.zeros_like(F_array)  # Phase G
    mape_array = np.full_like(F_array, np.nan)
    status_map = np.zeros(pol0_arr.shape, dtype=np.int8)
    fail_count = 0

    for res in results:
        i, j, F_fit, G_fit, status, mape = res
        row_idx = i - row_start
        col_idx = j - column_start

        F_array[row_idx, col_idx] = F_fit
        G_array[row_idx, col_idx] = G_fit
        mape_array[row_idx, col_idx] = mape
        status_map[i, j] = status
        if status == -1:
            fail_count += 1

    print(f'Sine fitting finished. Failed {fail_count} times.')

    x_coords = np.arange(column_start, column_end)
    y_coords = np.arange(row_start, row_end)
    x_array, y_array = np.meshgrid(x_coords, y_coords)

    np.save(subfolder_path / f'{image_name}_amplitude_array.npy', F_array)
    np.save(subfolder_path / f'{image_name}_phase_array.npy', G_array)
    np.save(subfolder_path / f'{image_name}_x_array.npy', x_array)
    np.save(subfolder_path / f'{image_name}_y_array.npy', y_array)


    # Visualise the fit status
    fig_status, ax_status = plt.subplots(1, 1, figsize=(8, 6))
    ax_status.imshow(pol0_img, cmap="gray")
    overlay = np.zeros((pol0_arr.shape[0], pol0_arr.shape[1], 4), dtype=np.float32)
    overlay[status_map == 1] = [0.0, 1.0, 0.0, 0.5]  # Success = Green
    overlay[status_map == -1] = [1.0, 0.0, 0.0, 0.5]  # Fail = Red
    ax_status.imshow(overlay)
    legend_elements = [Patch(facecolor='lime', alpha=0.5, label='Success'),
                       Patch(facecolor='red', alpha=0.5, label='Fail')]
    ax_status.legend(handles=legend_elements)
    ax_status.set_title("Fit Status on Image")
    ax_status.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(subfolder_path / f'{image_name}_sine_fit_summary_mp.png', bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close(fig_status)

    # Visualise error (MAPE) as an overlay
    avg_mape = np.nanmean(mape_array)
    fig_err, ax_err = plt.subplots(figsize=(8, 6))
    ax_err.imshow(pol0_img, cmap="gray")

    masked_mape = np.ma.masked_invalid(mape_array)

    # Only plot if there are valid MAPE values
    if np.any(masked_mape.mask == False):
        levels = np.linspace(np.nanmin(mape_array), np.nanmax(mape_array), 50)
        contour = ax_err.contourf(
            x_array, y_array, masked_mape,
            cmap='coolwarm', levels=levels,
            extend='both', alpha=1.0
        )
        cbar = fig_err.colorbar(contour, ax=ax_err, shrink=0.8)
        cbar.set_label('Mean Absolute Percentage Error (MAPE) %')

    fail_overlay = np.zeros((pol0_arr.shape[0], pol0_arr.shape[1], 4), dtype=np.float32)
    fail_overlay[status_map == -1] = [0.0, 0.0, 0.0, 0.6]  # Black overlay for fails
    ax_err.imshow(fail_overlay)

    ax_err.legend(handles=[Patch(facecolor='black', alpha=0.6, label='Fail')])
    ax_err.set_title(f'Sine Fit Error (MAPE) Overlay\nAverage MAPE: {avg_mape:.2f}%, Failed {fail_count} times')
    ax_err.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(subfolder_path / f'{image_name}_sine_fit_error_mp.png', bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close(fig_err)

    print("MAPE error overlay plot created.")
    print("Processing complete.")
