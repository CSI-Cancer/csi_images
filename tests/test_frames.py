import cv2

from csi_images import csi_frames, csi_tiles, csi_scans

SHOW_PLOTS = False


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


def test_making_composite_frames():
    scan = csi_scans.Scan.load_txt("tests/data")
    tile = csi_tiles.Tile(scan, 1000)
    frames = csi_frames.Frame.get_frames(tile)

    if SHOW_PLOTS:
        for frame in frames:
            cv2.imshow("Frames from a tile", frame.get_image()[0])
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    channel_indices = scan.get_channel_indices(["TRITC", "CY5", "DAPI"])
    image = csi_frames.make_rgb_image(tile, channel_indices)
    assert image.shape == (scan.tile_height_px, scan.tile_width_px, 3)

    if SHOW_PLOTS:
        cv2.imshow("RGB tile", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Test with a white channel
    channel_indices = scan.get_channel_indices(["TRITC", "CY5", "DAPI", "AF488"])
    image = csi_frames.make_rgbw_image(tile, channel_indices)
    assert image.shape == (scan.tile_height_px, scan.tile_width_px, 3)

    if SHOW_PLOTS:
        cv2.imshow("RGBW tile", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
