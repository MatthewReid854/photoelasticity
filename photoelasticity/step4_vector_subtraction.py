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


import numpy as np
from pathlib import Path

def vector_subtraction(directory_name, unstrained_image_name, strained_image_name):
    """
    Calculates the vector difference between strained and unstrained data.

    This function implements the Photoelastic Vector Subtraction Technique.
    It loads amplitude and phase data (polar coordinates) for both the strained and unstrained states from their
    respective NumPy files.

    To perform the subtraction, it first converts both vector fields into their Cartesian (x, y) components. The
    subtraction is then carried out on these components. The resulting difference vector field is converted back to
    polar coordinates (final amplitude and phase) and saved to new files.

    Parameters
    ----------
    directory_name : str
        The subfolder containing the input arrays and where the output arrays will be saved.
    unstrained_image_name : str
        The base name used to load the unstrained amplitude and phase `.npy` files.
    strained_image_name : str
        The base name used to load the strained amplitude and phase `.npy` files.

    Returns
    -------
    None

    Side Effects
    ------------
    - Saves two NumPy array files (`.npy`) to the specified directory:
        - `final_amplitude_array.npy`
        - `final_phase_array.npy`
    """

    current_directory = Path.cwd()
    subfolder_path = current_directory / directory_name

    # Load the vector arrays
    amp_strained = np.load(subfolder_path / f'{strained_image_name}_amplitude_array.npy')
    phase_strained_rad = np.load(subfolder_path / f'{strained_image_name}_phase_array.npy')
    amp_unstrained = np.load(subfolder_path / f'{unstrained_image_name}_amplitude_array.npy')
    phase_unstrained_rad = np.load(subfolder_path / f'{unstrained_image_name}_phase_array.npy')

    # 1. Convert all polar vectors to Cartesian coordinates at once
    x_strained = amp_strained * np.cos(phase_strained_rad)
    y_strained = amp_strained * np.sin(phase_strained_rad)

    x_unstrained = amp_unstrained * np.cos(phase_unstrained_rad)
    y_unstrained = amp_unstrained * np.sin(phase_unstrained_rad)

    # 2. Perform the subtraction on the Cartesian arrays
    x_final = x_strained - x_unstrained
    y_final = y_strained - y_unstrained

    # 3. Convert the resultant Cartesian array back to polar coordinates
    final_amplitude = np.sqrt(x_final**2 + y_final**2)
    final_phase_rad = np.arctan2(y_final, x_final) # Result is in radians

    # 4. Save the final arrays
    np.save(subfolder_path / 'final_amplitude_array.npy', final_amplitude)
    np.save(subfolder_path / 'final_phase_array.npy', final_phase_rad)
    print('Vector subtraction complete')
