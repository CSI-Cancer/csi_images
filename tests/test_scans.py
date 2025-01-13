import os

from csi_images.csi_scans import Scan


def test_from_yaml():
    scan = Scan.load_yaml("tests/data")
    # Assert that the scan is valid
    assert scan.slide_id != ""
    assert scan.exists
    assert scan.path != ""
    assert scan.start_datetime != ""
    assert scan.end_datetime != ""
    assert scan.scan_time_s > 0
    assert scan.scanner_id != ""
    assert scan.tray_pos != ""
    assert scan.slide_pos != ""
    assert scan.camera != ""
    assert scan.objective != ""
    assert scan.pixel_size_um > 0
    assert scan.tile_width_px > 0
    assert scan.tile_height_px > 0
    assert scan.tile_x_offset_px >= 0
    assert scan.tile_y_offset_px >= 0
    assert scan.tile_overlap_proportion >= 0
    assert len(scan.channels) > 0
    for channel in scan.channels:
        assert channel.name != ""
        assert channel.exposure_ms > 0
        assert channel.intensity > 0
        assert channel.gain_applied
    assert len(scan.roi) > 0
    for roi in scan.roi:
        assert roi.origin_x_um != -1
        assert roi.origin_y_um != -1
        assert roi.width_um > 0
        assert roi.height_um > 0
        assert roi.tile_rows > 0
        assert roi.tile_cols > 0
        assert roi.focus_points is not None

    # Should be able to read and autopopulate scan.yaml component
    assert scan == Scan.load_yaml("tests/data/scan.yaml")

    # Write, read, then delete the scan.yaml
    scan.save_yaml("tests/data/temp.yaml")
    assert scan == Scan.load_yaml("tests/data/temp.yaml")
    os.remove("tests/data/temp.yaml")

    # If we change the name, it's no longer equal but has the same profile
    scan2 = Scan.load_yaml("tests/data")
    scan2.slide_id = "DIFFERENT"
    assert scan != scan2
    assert scan.has_same_profile(scan2)

    # If we change the tile size, it no longer has the same profile
    scan2 = Scan.load_yaml("tests/data")
    scan2.tile_width_px = 1
    assert scan != scan2
    assert not scan.has_same_profile(scan2)


def test_from_txt():
    scan = Scan.load_txt("tests/data")
    # Assert that the scan is valid
    assert scan.slide_id != ""
    assert scan.exists
    assert scan.path != ""
    assert scan.start_datetime != ""
    assert scan.end_datetime != ""
    assert scan.scan_time_s > 0
    assert scan.scanner_id != ""
    assert scan.tray_pos != ""
    assert scan.slide_pos != ""
    # assert scan.camera != ""  # Unclear
    assert scan.objective != ""
    assert scan.pixel_size_um > 0
    assert scan.tile_width_px > 0
    assert scan.tile_height_px > 0
    assert scan.tile_x_offset_px >= 0
    assert scan.tile_y_offset_px >= 0
    assert scan.tile_overlap_proportion >= 0
    assert len(scan.channels) > 0
    for channel in scan.channels:
        assert channel.name != ""
        assert channel.exposure_ms > 0
        assert channel.intensity > 0
        # assert channel.gain_applied  # Not necessarily clear
    assert len(scan.roi) > 0
    for roi in scan.roi:
        assert roi.origin_x_um != -1
        assert roi.origin_y_um != -1
        assert roi.width_um > 0
        assert roi.height_um > 0
        assert roi.tile_rows > 0
        assert roi.tile_cols > 0
        assert roi.focus_points is not None

    # Should be able to read and autopopulate scan.txt component
    assert scan == Scan.load_txt("tests/data/slideinfo.txt")


def test_dict():
    scan = Scan.load_txt("tests/data")
    # Should be the same back and forth from dictionary
    assert scan == Scan.from_dict(scan.to_dict())


def test_names_and_indices():
    # Should be able to get the correct indices for the channels
    scan = Scan.load_txt("tests/data")
    correct_channel_order = ["DAPI", "TRITC", "CY5", "BF", "FITC"]
    assert scan.get_channel_indices(correct_channel_order) == [0, 1, 2, 3, 4]

    # Should return -1 for None
    assert scan.get_channel_indices([None]) == [-1]

    # Should return -1 for invalid channel names
    assert scan.get_channel_indices(["INVALID"]) == [-1]


def test_image_size():
    scan = Scan.load_yaml("tests/data")
    assert scan.get_image_size() == (2020, 2020)
