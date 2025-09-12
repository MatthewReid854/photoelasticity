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

import matplotlib.pyplot as plt
from step1_split_polarisations import process_polarized_image
# from step2_image_transformation import get_alignment_points, transform_images
from step3_fit_sine import fit_sine_functions
from step4_vector_subtraction import vector_subtraction
from step5_plots import contour_plots, draw_bounding_box, quiver_plot

if __name__ == '__main__':

    show_plots = False

    directory_name = 'test1' # the directory to save all the images and plots
    unstrained_image_name = 'UNSTRAINED'
    strained_image_name = 'STRAINED'

    input_path_unstrained=rf'D:\PVST\UNSTRAINED.tiff'
    input_path_strained=rf'D:\PVST\STRAINED.tiff'
    # image_path_black = rf'D:\PVST\BLACK.tiff' # this is the image with the lens cap on

    process_polarized_image(input_image_path=input_path_unstrained, directory_name=directory_name, show_plot=show_plots)
    process_polarized_image(input_image_path=input_path_strained, directory_name=directory_name, show_plot=show_plots)

    ## If image alignment is required, uncomment this section
    # get_alignment_points(directory_name,image_name=unstrained_image_name)
    # get_alignment_points(directory_name,image_name=strained_image_name)
    ## alignment points need to be entered manually after getting the alignment points using the above lines
    # unstrained_points = [(41, 170), (574, 152), (114, 295), (591, 308)]
    # strained_points = [(41, 170), (574, 153), (114, 295), (591, 309)]
    # transform_images(directory_name, unstrained_image_name, strained_image_name, unstrained_points, strained_points, show_plot=True)

    ## Bounding box dimensions
    bbox_column_start = 2
    bbox_row_start = 160
    width = 600
    height = 140
    bbox = (bbox_row_start, bbox_row_start+height+1, bbox_column_start, bbox_column_start+width+1)
    ## if you want to see the bounding box before processing the images (highly recommended the first time) then uncomment the following line
    # draw_bounding_box(directory_name, strained_image_name=strained_image_name, unstrained_image_name=unstrained_image_name, bounding_box=bbox)

    fit_sine_functions(directory_name, image_name=unstrained_image_name, bounding_box=bbox, show_plot = show_plots)
    fit_sine_functions(directory_name, image_name=strained_image_name, bounding_box=bbox, show_plot = show_plots)
    vector_subtraction(directory_name, unstrained_image_name=unstrained_image_name, strained_image_name=strained_image_name)

    contour_plots(directory_name=directory_name, strained_image_name=strained_image_name, bounding_box=bbox, gaussian_sigma=3, mask_edges=True, hide_colorbar=True, log_scale=False, only_plot_amplitude=True, show_plot=True, cmap='coolwarm')

    ## The following lines contain other plots that can be uncommented as required
    # quiver_plot(directory_name, strained_image_name, gaussian_sigma=3, bounding_box=bbox, mask_features=True, skip_rows=10, skip_cols=2, scale=0.005, cmap='viridis', show_plot=True)
    # correlation_plot(directory_name=directory_name, bounding_box=bbox, strained_image_name=strained_image_name, unstrained_image_name=unstrained_image_name)
    # vector_plot(directory_name, unstrained_image_name, strained_image_name, plot_type='cartesian')
    # vector_plot(directory_name, unstrained_image_name, strained_image_name, plot_type='polar')
    # PDF_plot(directory_name=directory_name, bounding_box=bbox, image_name=strained_image_name, combined=False)
    # histogram_plots(directory_name=directory_name, image_name=strained_image_name, ln=False)
    # intensity_violins(directory_name, bounding_box = bbox, image_name=strained_image_name)
    # plot_intensities(directory_sup_name='PEC_angle_', angles=['-50','-40','-30','-20','-10','0','10','20','30','40','50'], bounding_box=bbox)

    ## To plot multiple contour plots together, you can pass the ax to contour_plots. The blow example changes gaussian_sigma
    # fig, ax = plt.subplots(2, 3, figsize=(10, 8))
    # axes = ax.flatten()
    # for i in range(6):
    #     contour_plots(directory_name=directory_name, strained_image_name=strained_image_name,bounding_box=bbox, gaussian_sigma=i, mask_edges=True, ax=axes[i], hide_colorbar=True, log_scale=False, only_plot_amplitude=True, show_plot=False)
    #     axes[i].set_title("Gaussian Sigma = "+str(i))
    # plt.show()