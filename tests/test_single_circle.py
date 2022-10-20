from pathlib import Path
from tempfile import mkdtemp

from symmetric_synthetic.shape_generate import CreatePattern


def test_image_created():

    tpath = Path(mkdtemp())
    images_ = tpath / "images"
    images_.mkdir()

    masks_ = tpath / "masks"
    masks_.mkdir()

    image, mask = CreatePattern(
        no_of_images=1,
        # save_folder=str(images_),
        save_folder=str(tpath),
        pattern="triangle",
    )

    assert image.shape

    for c in tpath.iterdir():
        c.unlink()
