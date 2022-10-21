import shutil
from pathlib import Path
from tempfile import mkdtemp

from symmetric_synthetic.shape_generate import CreatePattern


def test_images_created():

    tpath = Path(mkdtemp())
    images_ = tpath / "images"
    images_.mkdir()

    masks_ = tpath / "masks"
    masks_.mkdir()

    images, masks = CreatePattern(
        no_of_images=1,
        # save_folder=str(images_),
        save_folder=str(tpath),
        pattern="circle",
    )

    assert len(images) > 0
    assert len(masks) > 0

    nimages = len([it for it in images_.iterdir()])
    nmasks = len([it for it in masks_.iterdir()])

    assert nimages > 0
    assert nmasks > 0

    assert nimages == 1
    assert nmasks == 1

    shutil.rmtree(tpath)


def test_image_object_created():

    tpath = Path(mkdtemp())
    images_ = tpath / "images"
    images_.mkdir()

    masks_ = tpath / "masks"
    masks_.mkdir()

    image, mask = CreatePattern(
        no_of_images=1,
        pattern="circle",
        f_save=False,
    )

    assert image.size[0] > 0
    assert image.size[1] > 0
    assert image.size[1] == image.size[0]

    shutil.rmtree(tpath)
