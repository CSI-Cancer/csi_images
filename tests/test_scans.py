import os
from csi_images import csi_scans


def test_from_yaml():
    # Should be able to read and autopopulate scan.yaml component
    scan = csi_scans.Scan.load_yaml("tests/data")
    scan2 = csi_scans.Scan.load_yaml("tests/data/scan.yaml")
    assert scan == scan2

    # Write, read, then delete the scan.yaml
    scan.save_yaml("tests/data/temp.yaml")
    scan3 = csi_scans.Scan.load_yaml("tests/data/temp.yaml")
    assert scan == scan3
    os.remove("tests/data/temp.yaml")


def test_from_txt():
    # Should be able to read and autopopulate scan.txt component
    scan = csi_scans.Scan.load_txt("tests/data")
    scan2 = csi_scans.Scan.load_txt("tests/data/slideinfo.txt")
    assert scan == scan2


def test_names_and_indices():
    # Should be able to get the correct indices for the channels
    scan = csi_scans.Scan.load_txt("tests/data")
    correct_channel_order = ["DAPI", "TRITC", "CY5", "FITC"]
    assert scan.get_channel_indices(correct_channel_order) == [0, 1, 2, 3]

    # Should return -1 for None
    assert scan.get_channel_indices([None]) == [-1]

    # Should raise an error if the channel is not found
    try:
        scan.get_channel_indices(["DAPI", "TRITC", "CY5", "FITC", "INVALID"])
        assert False
    except ValueError:
        assert True
