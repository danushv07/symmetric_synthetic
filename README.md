<!--
SPDX-FileCopyrightText: 2022 Venkatesh Danush Kumar <Danush-Kumar.Venkatesh@student.tu-freiberg.de>, Peter Steinbach <p.steinbach@hzdr.de>

SPDX-License-Identifier: BSD-3-Clause-Attribution
-->

# Shape generate

This package has the functionality to generate symmetrical objects such as circle, ellipses, square and triangles in an image and its corresponding mask. A combination of these objects can also be created. The function within creates two folders, namely, ```images``` and ```masks```. The images are constructed from the Gaussain distribution.

# Installation

This repo is managed with `poetry`. To try out this package, do:

```
$ poetry install
$ poetry run symmsyn --help
```

# Examples

The following examples show the method to create the dataset

1. Create a dataset of `200` images of size=256 x 256 with circle objects of radius=`10` and density=`20`
```bash
poetry run symmsyn --job_type "single" --n_image 200 --folder_name "circle_data" --object_type "circle" --image_dims 256 --circle_radius 10 --start_count 45 --end_count 50
```

2. Create a dataset of `100` images of size=512 x 512 with square objects of side=`10`, density=`35`
```bash
poetry run symmsyn --job_type "single" --n_image 100 --folder_name "square_data" --object_type "square" --image_dims 512 --start_count 34 --end_count 35 --square_size 10 10
```

3. Create a dataset of `100` images of size=512 x 512 with triangle objects of varying size, density=`35`
```bash
poetry run symmsyn --job_type "single" --n_image 100 --folder_name "triangle_data" --object_type "triangle" --image_dims 512 --start_count 34 --end_count 35 --size_vary True
```

4. Create a dataset of `500` images of size=256 x 256 with circle+ellipse objects of radius=`10`, minor_radius=`5`, major_radius=`10`, circle density=`35`, ellipse density=`10`

The elliptical objects will be equally divided into horizontal and angular objects. Specify ```angle``` parameter if needed. 
```bash
poetry run symmsyn --job_type "multi" --n_image 100 --folder_name "circle_ellipse_data" --image_dims 256 --circle_radius 10 --circle_start 34 --circle_end 35 --add_ellipse True --ell_count 10 --m_radius 10 --n_radius 5 --ell_angle 45
```

5. Create a dataset of `500` images of size=256 x 256 with circle+ellipse+triangle objects of varying sizes and circle density=`10`, ellipse density=`10`, triangle density=`40`

The elliptical objects will be equally divided into horizontal and angular objects. Specify ```angle``` parameter if needed. 
```bash
poetry run symmsyn --job_type "multi" --n_image 500 --folder_name "circle_ellipse_tri_data" --image_dims 256 --circle_start 9 --circle_end 10 --add_tri True --add_ellipse True --tri_count 40 --ell_count 10 --size_vary True
```

6. Create a dataset of `500` images of size=256 x 256 with circle+square+triangle objects of radius=`10`, side=`10`, size=`12` and circle density=`10`, square density=`10`, triangle density=`15` and change the intensity of objects to `0.5*std.` of the background

To change from square to rectangle objects vary the values in the parameter ```size```, a good starting value (-8,12) 
```bash
poetry run symmsyn --job_type "multi" --n_image 500 --folder_name "circle_square_tri_data" --image_dims 256 --circle_radius 10 --circle_start 9 --circle_end 10 --add_square True --add_tri True --square_size 10 10 --tri_size 12 12 --sq_count 10 --tri_count 15 --intensity_vary True --intensity_ratio 0.5
```

7. Create only an image and mask size=256 x 256 with circle+square+triangle objects of radius=10,side=10, size=12 and circle density=10, square density=10, triangle density=15 

The image and mask are saved as a file "multi.png"

To change from square to rectangle objects vary the values in the parameter ```size```, a good starting value (-8,12) 
```bash
poetry run symmsyn --job_type "multi" --image_dims 256 --circle_radius 10 --circle_start 9 --circle_end 10 --add_square True --add_tri True --square_size 10 10 --tri_size 12 12 --sq_count 10 --tri_count 15 --save_file False
```
=================================================================================================================================================================

Some **example images** are shown here.


![alt text](https://github.com/danushv07/symmetric_synthetic/blob/main/images/initial_dataset.png)

The datasets in the order from left to right: circle objects, circle and ellipse, circle and square, circle, ellipse and triangle objects.

An example of circle, ellipse and triangle object dataset with the variation of signal to noise.

![alt text](https://github.com/danushv07/symmetric_synthetic/blob/main/images/noisy_dataset.png)
