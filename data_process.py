# SPDX-FileCopyrightText: 2022 Venkatesh Danush Kumar <Danush-Kumar.Venkatesh@student.tu-freiberg.de>, Peter Steinbach <p.steinbach@hzdr.de>
#
# SPDX-License-Identifier: BSD-3-Clause-Attribution

"""This files contains the data curation steps needed to create datasets in this project
   Author: Danush Kumar Venkatesh
"""

import torch
from shape_generate import *
import os
import matplotlib.pyplot as plt
import click
#from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@click.command()
@click.option('-jt','--job_type', type=str, default="single",
              help="the type of objects to be created", show_default=True)
@click.option('-ni','--n_image', type=int, default=10,
              help="the number of images and mask to create", show_default=True)
@click.option('-fl','--folder_name', type=str, default="objects",
              help="the folder name to save images/masks")
@click.option('-ob','--object_type', type=str, default="circle",
              help="the type of pattern tot create - circle,triangle,square", show_default=True)
@click.option('-id','--image_dims', type=int, default=256,
              help="the size of images and mask to create", show_default=True)
@click.option('-cr','--circle_radius', type=int, default=6,
              help="the radius of circle in circle object type", show_default=True)
@click.option('-st','--start_count', type=int, default=40,
              help="the start count for no. of objects in the image", show_default=True)
@click.option('-en','--end_count', type=int, default=45,
              help="the end count for no. of objects in the image", show_default=True)
@click.option('-cs','--circle_start', type=int, default=45,
              help="the start count for no. of circle in the image- only for multi", show_default=True)
@click.option('-ce','--circle_end', type=int, default=50,
              help="the end count for no. of circle in the image- only for multi", show_default=True)
@click.option('-ss','--square_size', type=(int,int), default=(10,10),
              help="the size of squares in square object type", show_default=True)
@click.option('-ts','--tri_size', type=(int,int), default=(12,12),
              help="the size of triangle in triangle object type", show_default=True)
@click.option('-as','--add_square', type=bool, default=False,
              help="the flag to add square pattern in multi object type", show_default=True)
@click.option('-at','--add_tri', type=bool, default=False,
              help="the flag to add triangle pattern in multi object type", show_default=True)
@click.option('-ae','--add_ellipse', type=bool, default=False,
              help="the flag to add ellipse pattern in multi object type", show_default=True)
@click.option('-rc','--sq_count', type=int, default=20,
              help="the no. of square pattern in multi object type", show_default=True)
@click.option('-tc','--tri_count', type=int, default=20,
              help="the no. of triangle pattern in multi object type", show_default=True)
@click.option('-ec','--ell_count', type=int, default=20,
              help="the no. of ellipse pattern in multi object type", show_default=True)
@click.option('-me','--m_radius', type=int, default=8,
              help="the major radius of ellipse pattern in multi object type", show_default=True)
@click.option('-ne','--n_radius', type=int, default=5,
              help="the minor radius of ellipse pattern in multi object type", show_default=True)
@click.option('-ea','--ell_angle', type=int, default=45,
              help="the angle ellipse pattern in multi object type", show_default=True)
@click.option('-sv','--size_vary', type=bool, default=False,
              help="the flag to vary the size of different objects", show_default=True)
@click.option('-iv','--intensity_vary', type=bool, default=False,
              help="the flag to vary the intensity of different objects", show_default=True)
@click.option('-ir','--intensity_ratio', type=float, default=2.0,
              help="the value to vary the intensity of different objects", show_default=True)
@click.option('-sa','--save_file', type=bool, default=True,
              help="the flag to save images and masks", show_default=True)
def main(job_type, n_image, folder_name, object_type, image_dims, circle_radius, start_count,
         end_count, circle_start, circle_end, square_size, tri_size, add_square, add_tri, add_ellipse, 
         sq_count, tri_count, ell_count, m_radius, n_radius, ell_angle, size_vary, intensity_vary, 
         intensity_ratio, save_file):
    """
    The main function to create synthetic datasets with symmetrical objects
    
    - a folder with given folder name is created
        - two subfolders images and masks within contain the data
    - the no. of circles are created between the start_count and end_count
    - intensity_vary is available only in multi object type
    - if save_file is false a plot of the image and mask is saved either as single.png or multi.png
    """
  
    if job_type == "single":
        print(f"Image/masks of size {image_dims}x{image_dims} with {object_type} pattern is created.")
        
        if save_file:
            CreatePattern(no_of_images=n_image, save_folder=folder_name, pattern=object_type, i_size=image_dims, 
                      disk_radius=circle_radius, count_begin=start_count, count_end=end_count, size=square_size, 
                      tri_shape=tri_size, f_save=True, var_size=size_vary)
        else:
            image, mask = CreatePattern(no_of_images=n_image, save_folder=folder_name, pattern=object_type, i_size=image_dims, 
                      disk_radius=circle_radius, count_begin=start_count, count_end=end_count, size=square_size, 
                      tri_shape=tri_size, f_save=False, var_size=size_vary)
            
            fig, ax = plt.subplots(1,2)
            ax[0].imshow(image, cmap='gray')
            ax[1].imshow(mask)
            plt.savefig(f"{job_type}.png", bbox_inches="tight")
    
    elif job_type == "multi":
        print(f"Images/masks of size {image_dims}x{image_dims} with {job_type} pattern is created.")
        if save_file:
            CreateMixture(no_of_images=n_image, save_folder=folder_name, add_sq=add_square, add_tri=add_tri, 
                      add_ell=add_ellipse, i_size=image_dims, disk_radius=circle_radius, count_begin=circle_start,
                      count_end=circle_end, rect_count=sq_count, size=square_size, tri_count=tri_count, 
                      ellip_count=ell_count, major_radius=n_radius, minor_radius=m_radius, angle=ell_angle, 
                      tri_shape=tri_size, f_save=True, var_size=size_vary, change_intensity=intensity_vary, 
                      signal_ratio=intensity_ratio)
            
        else:
            image, mask = CreateMixture(no_of_images=n_image, save_folder=folder_name, add_sq=add_square, add_tri=add_tri, 
                          add_ell=add_ellipse, i_size=image_dims, disk_radius=circle_radius, count_begin=circle_start,
                          count_end=circle_end, rect_count=sq_count, size=square_size, tri_count=tri_count, 
                          ellip_count=ell_count, major_radius=m_radius, minor_radius=n_radius, angle=ell_angle, 
                          tri_shape=tri_size, f_save=False, var_size=size_vary, change_intensity=intensity_vary, 
                          signal_ratio=intensity_ratio)
            
            fig, ax = plt.subplots(1,2)
            ax[0].imshow(image, cmap='gray')
            ax[1].imshow(mask)
            plt.savefig(f"{job_type}.png", bbox_inches="tight")
        
    else:
        print(f"The {job_type} is not available, choose from single or multi")


if __name__ == "__main__":
    main()