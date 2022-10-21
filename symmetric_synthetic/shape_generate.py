# SPDX-FileCopyrightText: 2022 Venkatesh Danush Kumar <Danush-Kumar.Venkatesh@student.tu-freiberg.de>, Peter Steinbach <p.steinbach@hzdr.de>
#
# SPDX-License-Identifier: BSD-3-Clause-Attribution

"""
This file contains functions to generate various shapes and inturn
the synthetic data
Author: Danush Kumar Venkatesh
"""

import os
# import glob
from glob import glob

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

# install dependencies
from skimage.draw import disk, ellipse, polygon, rectangle
from skimage.transform import rotate
from skimage.util import random_noise


def ImageRescale(image):  # , random_gen
    """
    Customized function to add noise and rescale image between 0, 255

    - random gaussian noise followed by gaussain filter is applied

    - the final image is rescaled and converted to np.uint8
    Parameters
    ---------------
    image : np.ndarray of float64

    Return
    ---------------
    rescaled image of stype np.uint8
    """

    # add noise and apply gaussian filter
    # image *= random_gen.normal(loc=1.0, scale=0.1, size=image.shape)
    image = ndi.gaussian_filter(image, sigma=0.8)

    # scale and threshold images values between 0 and 255
    i_min, i_max = image.min(), image.max()
    thresh1 = (255) / (i_max - i_min)
    thresh2 = 255 - thresh1 * i_max
    image_ = (thresh1 * image + thresh2).astype(np.uint8)

    return image_


def CreatePattern(
    no_of_images,
    save_folder="objects",
    pattern="circle",
    i_size=256,
    disk_radius=6,
    count_begin=35,
    count_end=50,
    size=(10, 10),
    tri_shape=(10, 10),
    f_save=True,
    var_size=False,
):
    """
    Function to create circle/square/triangle shaped pattern
    images and mask

    - a gaussian based image is created and patterns are added
    - the image/mask is stored in the np.uint8 format as .png files
    - the image and mask are saved in 2 different folder

    NOTE: for triangle pattern change the count_begin/end

    Parameters
    ------------------
    no_of_images :  int
        The number of images to generate
    save_folder : str, [Optional], default - "objects'
        The folder to save the images
    pattern : str, [Optional], default - circle
        circle - to create circle pattern
        square - to creat square pattern
        triangle - to create triangle pattern
    i_size : int, [Optional], default - 256
        The size of the image/mask to generate
    disk_radius: int, [Optional], default - 6
        The radius of the disk
    count_begin : int, [Optional], default - 35
        The start count of number of objects to be generated
    count_end : int, [Optional], default - 50
        The end count of number of objects to be generated
        A random number between is chosen for the counts
    size : tuple, [Optional], default - (10,10)
        The extent of spread of square
    tri_shape : tuple, [Optional], default - (10,10)
        The min and max size of triangles to include
    f_save : bool, [Optional], default - True
        The option to save the images to folders
        False - return a single image and mask
    var_size : bool, [Optional], default - False
        To include variable size objects

    Return
    ------------------
    images directory : the images saved in np.uint8 format as .png files
    masks directory : the corresponding mask saved in np.uint8 format as .png files


    """
    # code adapted from https://scikit-image.org/docs/dev/auto_examples/features_detection
    image_size = i_size
    radius = disk_radius

    # create copies for variable size
    rad = radius
    r_size = size
    t_size = tri_shape

    # define the path to save
    if f_save:
        path = os.path.join(os.getcwd(), save_folder)
        if not os.path.exists(path):
            for name in ["images", "masks"]:
                os.makedirs(os.path.join(path, name))

    images = []
    masks = []
    for j in range(no_of_images):
        rng = np.random.default_rng()
        img = rng.normal(loc=0.5, scale=0.1, size=(image_size, image_size))
        mask = np.zeros((image_size, image_size), dtype=np.uint8)
        count = int(np.random.randint(count_begin, count_end, size=1))

        for i in range(count):
            center = tuple(
                map(tuple, (np.random.randint(0, image_size - 5, size=(1, 2))))
            )

            if pattern == "circle":
                if var_size:
                    radius = rad * abs(float(np.random.randn(1)))
                rr, cc = disk(center[0], radius, shape=img.shape)
            elif pattern == "square":
                if var_size:
                    n = np.random.randint(0, 15)
                    size = tuple([s + n for s in r_size])
                rr, cc = rectangle(center[0], extent=size, shape=img.shape)

            elif pattern == "triangle":
                if var_size:
                    n = np.random.randint(0, 15)
                    tri_shape = tuple([s + n for s in t_size])
                point, tri_side, tri_height = CreateTriangle(
                    (image_size, image_size), tri_shape, rng
                )
                indexes = polygon(
                    [
                        point[0],
                        point[0] - tri_height,
                        point[0],
                    ],
                    [
                        point[1],
                        point[1] + tri_side // 2,
                        point[1] + tri_side,
                    ],
                )

            else:
                print(
                    f"The pattern {pattern} mentioned is not implemented.Choose from circle, triangle and square"
                )
                break

            if not (pattern == "triangle"):
                img[rr, cc] = 2  #
                mask[rr, cc] = 1
            else:
                img[indexes] = 2
                mask[indexes] = 1

        # apply noise & rescale image
        final_img = ImageRescale(img)  # , rng

        # using PIL to save images
        image = Image.fromarray(final_img)
        mask_ = Image.fromarray(mask * 255)

        if f_save:
            oimage = save_folder + "/images/image" + str(j + 1) + ".png"
            omask = save_folder + "/masks/mask" + str(j + 1) + ".png"
            image.save(oimage)
            mask_.save(omask)

            images.append(oimage)
            masks.append(omask)

        else:
            if j == 0:
                return image, mask_

    return images, masks


def CreateMixture(
    no_of_images,
    save_folder,
    add_sq=False,
    add_tri=False,
    add_ell=False,
    i_size=256,
    disk_radius=6,
    count_begin=35,
    count_end=50,
    rect_count=10,
    size=(10, 10),
    tri_count=10,
    ellip_count=10,
    major_radius=5,
    minor_radius=8,
    angle=45,
    tri_shape=(10, 10),
    f_save=True,
    var_size=False,
    change_intensity=False,
    signal_ratio=1,
    count_percent=1.0,
):
    """
    Function to create a mixture of patterns

    - as the main pattern circle is created and additionally other patterns can be
        added
    - a gaussian based image is created and patterns are added
    - the image/mask is stored in the np.uint8 format as .png files
    - the image and mask are saved in 2 different folder
    - the vaiable size option is only available for circle, triangle, ellipse &
        not for square patterns

    Parameters
    ------------------
    no_of_images :  int
        The number of images to generate
    save_folder : str
        The folder to save the images
    add_sq : bool, [Optional], default - True
        To add square pattern
    add_tri : bool, [Optional], default - False
        To add triangle pattern
    add_ell : bool, [Optional], default - False
        To add ellipse pattern
    i_size : int, [Optional], default - 256
        The size of the image/mask to generate
    disk_radius: int, [Optional], default - 6
        The radius of the disk
    count_begin : int, [Optional], default - 35
        The start count of number of disks to be generated
    count_end : int, [Optional], default - 50
        The end count of number of disks to be generated
        A random number between is chosen for the counts
    rect_count : : int, [Optional], default - 10
        The count of number of square to be generated
    size : tuple, [Optional], default - (10,10)
        The extent of spread of square
    tri_count : : int, [Optional], default - 10
        The count of number of triangles to be generated
    ellip_count : : int, [Optional], default - 10
        The half of count cretes horizontal and other creates angled ellipse
    major_radius: int, [Optional], default - 5
        The major radius of the ellipse
    minor_radius: int, [Optional], default - 8
        The minor radius of the ellipse
    angle : int, [Optional], default - 45
        The angle of the ellipse
    tri_shape : tuple, [Optional], default - (10,10)
        The min and max size of triangles to include
    f_save : bool, [Optional], default - True
        The option to save the images to folders
        False - return a single image and mask
    var_size : bool, [Optional], default - False
        The option to vary the size of objects
    change_intensity : bool, [Optional], default - False
        The option to vary the intensity of objects in the image
    signal_ratio : int, [Optional], default - 1
        the intensity of the objects is computed by multiplying with the
        std. of the image + mean.

    Return
    ------------------
    images directory : the images saved in np.uint8 format as .png files
    masks directory : the corresponding mask saved in np.uint8 format as .png files
    """
    image_size = i_size
    radius = disk_radius
    center_num = image_size - 10

    # create copies for variable size
    rad = radius
    r_size = size
    t_size = tri_shape
    m_rad = major_radius
    n_rad = minor_radius

    # define the path to save
    if f_save:
        path = os.path.join(os.getcwd(), save_folder)
        if not os.path.exists(path):
            for name in ["images", "masks"]:
                os.makedirs(os.path.join(path, name))

    for j in range(no_of_images):
        rng = np.random.default_rng()
        img = rng.normal(loc=0.5, scale=0.1, size=(image_size, image_size))
        mask = np.zeros((image_size, image_size), dtype=np.uint8)

        if var_size:
            img = np.pad(img, (10, 10), "constant")
            mask = np.pad(mask, (10, 10), "constant")
            w, h = img.shape

        count = int(np.random.randint(count_begin, count_end, size=1))

        if change_intensity:
            pixel_value = np.mean(img) + signal_ratio * np.std(img)
        else:
            pixel_value = 1

        for i in range(count):
            if var_size:
                radius = rad * (float(np.random.normal(size=1)))
            center = tuple(
                map(tuple, (np.random.randint(0, image_size - 5, size=(1, 2))))
            )
            rr, cc = disk(center[0], radius, shape=(image_size, image_size))
            img[rr, cc] = pixel_value  #
            mask[rr, cc] = 1

        # TODO: add variable size for square objects
        if add_sq:
            # if (var_size and count_percent<1.0):
            #    rect_count_r = int(count_percent*rect_count)
            #    rect_count = abs(rect_count - rect_count_r)
            # for k in range(rect_count_r):
            #    n = np.random.randint(0,15)
            #    size = tuple([s+n for s in r_size])
            #    center = tuple(map(tuple, (np.random.randint(0, image_size-5, size=(1,2)))))
            #    rr, cc = rectangle(center[0], extent=size, shape=img.shape)
            #    img[rr,cc] = pixel_value #
            #    mask[rr,cc] = 1+i+k

            for l in range(rect_count):
                center = tuple(
                    map(tuple, (np.random.randint(0, image_size - 5, size=(1, 2))))
                )
                rr, cc = rectangle(center[0], extent=r_size, shape=img.shape)
                img[rr, cc] = pixel_value  #
                mask[rr, cc] = 1

        if add_tri:
            for k in range(tri_count):
                if var_size:
                    n = float(np.random.normal(size=1))
                    tri_shape = tuple([s + n for s in t_size])
                point, tri_side, tri_height = CreateTriangle(
                    (image_size, image_size), tri_shape, rng
                )
                indexes = polygon(
                    [
                        point[0],
                        point[0] - tri_height,
                        point[0],
                    ],
                    [
                        point[1],
                        point[1] + tri_side // 2,
                        point[1] + tri_side,
                    ],
                )

                img[indexes] = pixel_value
                mask[indexes] = 1

        if add_ell:
            e_count = int(ellip_count / 2)
            for k in range(e_count):
                if var_size:
                    major_radius = m_rad * (float(np.random.normal(size=1)))
                    minor_radius = n_rad * (float(np.random.normal(size=1)))
                center = tuple(
                    map(tuple, (np.random.randint(0, center_num, size=(1, 2))))
                )
                rr, cc = ellipse(center[0][0], center[0][1], major_radius, minor_radius)

                img[rr, cc] = pixel_value  #
                mask[rr, cc] = 1

            for s in range(e_count):
                center = tuple(
                    map(tuple, (np.random.randint(0, center_num, size=(1, 2))))
                )
                rr, cc = ellipse(
                    center[0][0], center[0][1], 5, 8, rotation=np.deg2rad(angle)
                )
                img[rr, cc] = pixel_value
                mask[rr, cc] = 1

        # apply noise & rescale image
        final_img = ImageRescale(img)  # , rng

        # using PIL to save images
        if var_size:
            image = Image.fromarray(final_img[10 : w - 10, 10 : h - 10])
            mask_ = Image.fromarray(mask[10 : w - 10, 10 : h - 10] * 255)
        else:
            image = Image.fromarray(final_img)
            mask_ = Image.fromarray(mask * 255)

        if f_save:
            image.save(save_folder + "/images/image" + str(j + 1) + ".png")
            mask_.save(save_folder + "/masks/mask" + str(j + 1) + ".png")

        elif (not f_save) and (j == 0):
            return image, mask_
            break


def EllipsePattern(
    no_of_images,
    save_folder,
    i_size=256,
    major_radius=5,
    minor_radius=8,
    real_count=30,
    count_begin=10,
    count_end=20,
    angle=45,
    f_save=True,
):
    """
    Function to create ellipse shaped pattern images and mask

    - a gaussian based image is created and ellipse patterns are added
    - the image/mask is stored in the np.uint8 format as .png files
    - the image and mask are saved in 2 different folder

    Parameters
    ------------------
    no_of_images :  int
        The number of images to generate
    save_folder : str
        The folder to save the images
    i_size : int, [Optional], default - 256
        The size of the image/mask to generate
    major_radius: int, [Optional], default - 5
        The major radius of the ellipse
    minor_radius: int, [Optional], default - 8
        The minor radius of the ellipse
    real_count : int, [Optional], default - 30
        The horizontal count of numberof ellipses to be generated
    count_begin : int, [Optional], default - 35
        The start count of number of ellipses to be generated
    count_end : int, [Optional], default - 50
        The end count of number of ellipses to be generated
        A random number between is chosen for the angle ellipse
    angle : int, [Optional], default - 45
        The angle of the ellipse
    f_save : bool, [Optional], default - True
        The option to save the images to folders
        False - return a single image and mask

    Return
    ------------------
    images directory : the images saved in np.uint8 format as .png files
    masks directory : the corresponding mask saved in np.uint8 format as .png files
    """
    # code adapted from https://scikit-image.org/docs/dev/auto_examples/features_detection
    image_size = i_size
    center_num = image_size - 11  # safe value to avoid out of bounds error

    # define the path to save
    if f_save:
        path = os.path.join(os.getcwd(), save_folder)
        if not os.path.exists(path):
            for name in ["images", "masks"]:
                os.makedirs(os.path.join(path, name))

    for j in range(no_of_images):
        rng = np.random.default_rng()
        img = rng.normal(loc=0.5, scale=0.1, size=(image_size, image_size))
        mask = np.zeros((image_size, image_size), dtype=np.uint8)
        count = int(np.random.randint(10, 20, size=1))

        for i in range(real_count):
            center = tuple(map(tuple, (np.random.randint(0, center_num, size=(1, 2)))))
            rr, cc = ellipse(center[0][0], center[0][1], major_radius, minor_radius)
            img[rr, cc] = 1
            mask[rr, cc] = 1 + i

        for k in range(count):
            center = tuple(map(tuple, (np.random.randint(0, center_num, size=(1, 2)))))
            rr, cc = ellipse(
                center[0][0], center[0][1], 5, 8, rotation=np.deg2rad(angle)
            )
            img[rr, cc] = 1
            mask[rr, cc] = 1 + i + k

        # apply noise & rescale image
        final_img = ImageRescale(img)  # , rng

        # using PIL to save images
        image = Image.fromarray(final_img)
        mask_ = Image.fromarray(mask * 255)

        if f_save:
            image.save(save_folder + "/images/image" + str(j + 1) + ".png")
            mask_.save(save_folder + "/masks/mask" + str(j + 1) + ".png")

        elif (not f_save) and (j == 0):
            return image, mask_
            break


def CreateTriangle(img_shape, shape, rng):
    """
    Aux. func. to create the indices for the traingle pattren

    Parameters
    --------------
    img_shape : tuple
        The shape of the input image
    shape : tuple
        The min. and max. size of the triangles
    random_gen: np.randomgenerator

    Return
    -------------
    st_point : tuple, (row coord., column coord.)
        The starting point of the traingle vertex
    triangle_side :  int
        The side length of the triangle
    height : int
        The height of the triangle
    """

    # code has been adapted from
    # https://github.com/scikit-image/scikit-image/blob/main/skimage/draw/_random_shapes.py
    row = rng.integers(max(1, img_shape[0] - shape[0]))
    colm = rng.integers(max(1, img_shape[1] - shape[0]))
    st_point = (row, colm)
    calc_side = min(img_shape[1] - st_point[1], st_point[0], shape[1]) - shape[0]
    triangle_side = shape[0] + rng.integers(max(1, calc_side)) - 1
    height = int(np.ceil(np.sqrt(3 / 4.0) * triangle_side))

    return st_point, triangle_side, height


def PatternInv(image_path, mask_path, folder, f_save=True):
    """
    Function to creta inversions patterns of image

    Parameters
    ------------------------
    image_path : str
        The name of path to the images
    mask_path : str
        The name of path to the masks
    folder : str
        The folder name to save images and masks

    Return
    ------------------------
    The flipped and inversion version of images & masks
    saved in the given folder name
    """

    # open and read the dataset
    X_img = sorted(glob(image_path + "/*.png"))
    X_mask = sorted(glob(mask_path + "/*.png"))

    length = len(X_img)
    save_folder = folder

    X = list(map(Image.open, X_img))
    Y = list(map(Image.open, X_mask))

    j = 0
    for image, mask in zip(X, Y):

        # create flip image
        img_flip = np.flip(np.asarray(image), axis=1)
        img_lr = np.fliplr(np.flip((np.asarray(image)).T, axis=0))
        img_lr2 = np.fliplr(np.flip((img_flip), axis=0).T)

        # create flip mask
        mask_flip = np.flip(np.asarray(mask), axis=1)
        mask_lr = np.fliplr(np.flip((np.asarray(mask)).T, axis=0))
        mask_lr2 = np.fliplr(np.flip((mask_flip), axis=0).T)

        # concat image
        img_part1 = np.concatenate([image, img_flip], axis=1)
        img_part2 = np.concatenate([img_lr2, img_lr], axis=1)
        final_image = np.concatenate([img_part1, img_part2], axis=0)

        # concat mask
        mask_part1 = np.concatenate([mask, mask_flip], axis=1)
        mask_part2 = np.concatenate([mask_lr2, mask_lr], axis=1)
        final_mask = np.concatenate([mask_part1, mask_part2], axis=0)

        # using PIL to save images
        image_ = Image.fromarray(final_image)
        mask_ = Image.fromarray(final_mask)

        # save image & mask
        if f_save:
            image_.save(save_folder + "/images_inv/image" + str(j + 1) + ".png")
            mask_.save(save_folder + "/masks_inv/mask" + str(j + 1) + ".png")

        j += 1
