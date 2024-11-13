import os

import cv2

from csi_images import csi_frames, csi_tiles, csi_scans

if os.environ.get("DEBIAN_FRONTEND") == "noninteractive":
    SHOW_PLOTS = False
else:
    # Change this to your preference for local testing, but commit as True
    SHOW_PLOTS = True


def test_getting_frames():
    scan = csi_scans.Scan.load_yaml("tests/data")
    tile = csi_tiles.Tile(scan, 100)
    frames = csi_frames.Frame.get_frames(tile)
    assert len(frames) == 4
    frames = csi_frames.Frame.get_all_frames(scan)
    assert len(frames) == scan.roi[0].tile_rows * scan.roi[0].tile_cols
    assert len(frames[0]) == 4
    frames = csi_frames.Frame.get_all_frames(scan, as_flat=False)
    assert len(frames) == scan.roi[0].tile_rows
    assert len(frames[0]) == scan.roi[0].tile_cols
    assert len(frames[0][0]) == 4


def test_checking_frames():
    scan = csi_scans.Scan.load_yaml("tests/data")
    tile = csi_tiles.Tile(scan, 100)
    frames = csi_frames.Frame.get_frames(tile)
    assert len(frames) == 4
    for frame in frames:
        assert frame.check_image()
    assert csi_frames.Frame.check_all_images(scan)
    # Manually set up a frame that shouldn't exist
    tile.x = 100
    for frame in csi_frames.Frame.get_frames(tile):
        assert not frame.check_image()


def test_make_rgb():
    scan = csi_scans.Scan.load_txt("tests/data")
    tile = csi_tiles.Tile(scan, 1000)
    frames = csi_frames.Frame.get_frames(tile)

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
    image = csi_frames.Frame.make_rgb_image(tile, channels)
    assert image.shape == (scan.tile_height_px, scan.tile_width_px, 3)

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
    image = csi_frames.Frame.make_rgb_image(tile, channels)
    assert image.shape == (scan.tile_height_px, scan.tile_width_px, 3)

    if SHOW_PLOTS:
        cv2.imshow("RGBW tile", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
