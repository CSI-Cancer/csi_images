import os

import pytest

import cv2

from csi_images.csi_scans import Scan
from csi_images.csi_tiles import Tile
from csi_images.csi_frames import Frame

if os.environ.get("DEBIAN_FRONTEND") == "noninteractive":
    # For Docker testing; do not try to show plots
    SHOW_PLOTS = False
else:
    # Change this to your preference for local testing, but commit as True
    SHOW_PLOTS = True


@pytest.fixture
def scan():
    return Scan.load_yaml("tests/data")


@pytest.fixture
def tile(scan):
    return Tile(scan, 100)


def test_paths(scan, tile):
    frames = Frame.get_frames(tile)
    # Check that the paths exist
    for frame in frames:
        # Manually creating the path
        file_path = os.path.join(scan.path, frame.get_file_name())
        assert os.path.exists(file_path)
        # Using the build-in method
        assert os.path.exists(frame.get_file_path())
        # Check that the images are "valid"
        assert frame.check_image()
    # Check that all images are valid
    assert Frame.check_all_images(scan)
    # Manually set up a frame that shouldn't exist
    tile.x = 100
    for frame in Frame.get_frames(tile):
        assert not frame.check_image()


def test_getting_frames(scan, tile):
    # Get frames for a single tile
    frames = Frame.get_frames(tile)
    assert len(frames) == 4

    # Get all frames for the scan
    frames = Frame.get_all_frames(scan)
    assert len(frames) == scan.roi[0].tile_rows * scan.roi[0].tile_cols
    assert len(frames[0]) == 4
    # Get all frames in a grid
    frames = Frame.get_all_frames(scan, as_flat=False)
    assert len(frames) == scan.roi[0].tile_rows
    assert len(frames[0]) == scan.roi[0].tile_cols
    assert len(frames[0][0]) == 4


def test_make_rgb(scan, tile):
    frames = Frame.get_frames(tile)

    if SHOW_PLOTS:
        for frame in frames:
            cv2.imshow("Frames from a tile", frame.get_image())
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    channel_indices = scan.get_channel_indices(["TRITC", "CY5", "DAPI"])
    channels = {
        channel_indices[0]: (1.0, 0.0, 0.0),
        channel_indices[1]: (0.0, 1.0, 0.0),
        channel_indices[2]: (0.0, 0.0, 1.0),
    }
    image = Frame.make_rgb_image(tile, channels)
    real_image_size = scan.get_image_size() + (3,)
    assert image.shape == real_image_size

    if SHOW_PLOTS:
        cv2.imshow("RGB tile", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Test with a white channel
    channel_indices = scan.get_channel_indices(["TRITC", "CY5", "DAPI", "AF488"])
    channels = {
        channel_indices[0]: (1.0, 0.0, 0.0),
        channel_indices[1]: (0.0, 1.0, 0.0),
        channel_indices[2]: (0.0, 0.0, 1.0),
        channel_indices[3]: (1.0, 1.0, 1.0),
    }
    image = Frame.make_rgb_image(tile, channels)
    assert image.shape == real_image_size

    if SHOW_PLOTS:
        cv2.imshow("RGBW tile", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
