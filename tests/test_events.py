import os
import time

import cv2
import numpy as np
import pandas as pd

from csi_images.csi_scans import Scan
from csi_images.csi_tiles import Tile
from csi_images.csi_events import Event, EventArray
from csi_images import csi_images

if os.environ.get("DEBIAN_FRONTEND") == "noninteractive":
    SHOW_PLOTS = False
else:
    # Change this to your preference for local testing, but commit as True
    SHOW_PLOTS = True


def test_getting_event():
    scan = Scan.load_txt("tests/data")
    tile = Tile(scan, 0)
    event = Event(
        scan,
        tile,
        516,
        278,
    )
    # Test getting images
    images = event.get_crops()
    assert len(images) == 4
    images = event.get_crops(crop_size=100)
    assert images[0].shape == (100, 100)
    images = event.get_crops(crop_size=50)
    assert images[0].shape == (50, 50)

    if SHOW_PLOTS:
        for image in images:
            cv2.imshow("Bright DAPI event in the center", image)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Test a corner event
    event = Event(
        scan,
        tile,
        1346,
        9,
    )
    images = event.get_crops()
    assert len(images) == 4
    images = event.get_crops(crop_size=100)
    assert images[0].shape == (100, 100)
    images = event.get_crops(crop_size=200)
    assert images[0].shape == (200, 200)

    if SHOW_PLOTS:
        for image in images:
            cv2.imshow("Events in the top-right corner of a tile", image)
            cv2.waitKey(0)
        cv2.destroyAllWindows()


def test_getting_many_events():
    scan = Scan.load_txt("tests/data")
    tile = Tile(scan, 1000)
    tile2 = Tile(scan, 0)
    events = [
        Event(scan, tile, 515, 411),
        Event(scan, tile2, 2, 1000),
        Event(scan, tile, 1000, 1000),
        Event(scan, tile, 87, 126),
        Event(scan, tile, 1000, 2),
        Event(scan, tile2, 800, 800),
        Event(scan, tile, 1000, 662),
    ]
    # Test time to extract images sequentially
    start_time = time.time()
    images_1 = []
    for event in events:
        images_1.append(event.get_crops())
    sequential_time = time.time() - start_time

    # Test time to extract images in parallel
    start_time = time.time()
    images_2 = Event.get_many_crops(events, crop_size=100)
    parallel_time = time.time() - start_time
    assert parallel_time < sequential_time
    for list_a, list_b in zip(images_1, images_2):
        assert len(list_a) == len(list_b)
        for image_a, image_b in zip(list_a, list_b):
            assert np.array_equal(image_a, image_b)

    # Test that it works after converting to EventArray and back
    event_array = EventArray.from_events(events)
    remade_events = event_array.to_events(
        [scan], ignore_metadata=True, ignore_features=True
    )
    images_3 = Event.get_many_crops(remade_events, crop_size=100)
    for list_a, list_b in zip(images_1, images_3):
        assert len(list_a) == len(list_b)
        for image_a, image_b in zip(list_a, list_b):
            assert np.array_equal(image_a, image_b)


def test_event_coordinates_for_bzscanner():
    scan = Scan.load_txt("tests/data")
    # Origin
    tile = Tile(scan, 0)
    event = Event(scan, tile, 0, 0)
    scan_origin = event.get_scan_position()
    assert 2500 <= scan_origin[0] <= 3500
    assert 2500 <= scan_origin[1] <= 3500
    scan_origin_on_slide = event.get_slide_position()
    assert 71500 <= scan_origin_on_slide[0] <= 72500
    assert 21500 <= scan_origin_on_slide[1] <= 22500
    # Within the same tile, "bottom-right corner"
    event = Event(scan, tile, 1000, 1000)
    scan_position = event.get_scan_position()
    assert scan_origin[0] <= scan_position[0]
    assert scan_origin[1] <= scan_position[1]
    slide_position = event.get_slide_position()
    assert slide_position[0] <= scan_origin_on_slide[0]
    assert slide_position[1] <= scan_origin_on_slide[1]

    # Next row, opposite side
    tile = Tile(scan, (scan.roi[0].tile_cols - 1, 1))
    event = Event(scan, tile, 1000, 1000)
    scan_position = event.get_scan_position()
    assert scan_origin[0] <= scan_position[0]
    assert scan_origin[1] <= scan_position[1]
    slide_position = event.get_slide_position()
    assert slide_position[0] <= scan_origin_on_slide[0]
    assert slide_position[1] <= scan_origin_on_slide[1]

    # Opposite corner
    tile = Tile(scan, (scan.roi[0].tile_cols - 1, scan.roi[0].tile_rows - 1))
    event = Event(scan, tile, 1361, 1003)
    scan_position = event.get_scan_position()
    assert 21500 <= scan_position[0] <= 22500
    assert 58500 <= scan_position[1] <= 60500
    slide_position = event.get_slide_position()
    assert 14500 <= slide_position[0] <= 15500
    assert 2500 <= slide_position[1] <= 3500


def test_event_coordinates_for_axioscan():
    scan = Scan.load_yaml("tests/data")
    # Origin
    tile = Tile(scan, 0)
    event = Event(scan, tile, 0, 0)
    scan_position = event.get_scan_position()
    assert -59000 <= scan_position[0] < -55000
    assert 0 <= scan_position[1] < 4000
    slide_position = event.get_slide_position()
    assert 16000 <= slide_position[0] < 20000
    assert scan_position[1] == slide_position[1]

    # Opposite corner
    tile = Tile(scan, (scan.roi[0].tile_cols - 1, scan.roi[0].tile_rows - 1))
    event = Event(scan, tile, 2000, 2000)
    scan_position = event.get_scan_position()
    assert -4000 <= scan_position[0] <= 0
    assert 21000 <= scan_position[1] <= 25000
    slide_position = event.get_slide_position()
    assert 71000 <= slide_position[0] <= 75000
    assert scan_position[1] == slide_position[1]


def test_eventarray_conversions():
    scan = Scan.load_yaml("tests/data")
    # Origin
    tile = Tile(scan, 0)
    event0 = Event(scan, tile, 0, 0)
    event1 = Event(scan, tile, 1000, 1000)
    event2 = Event(scan, tile, 2000, 2000)

    event_array = EventArray.from_events([event0, event1, event2])

    assert len(event_array) == 3
    assert event_array.metadata is None
    assert event_array.features is None

    event0.metadata = pd.Series({"event0": 0})

    try:
        event_array = EventArray.from_events([event0, event1, event2])
        # Should throw error
        assert False
    except ValueError:
        pass

    event1.metadata = pd.Series({"event0": 1})
    event2.metadata = pd.Series({"event0": 2})

    event_array = EventArray.from_events([event0, event1, event2])

    assert len(event_array) == 3

    events_df = event_array.to_dataframe()

    assert len(events_df) == 3

    assert event_array == EventArray.from_dataframe(events_df)

    # Test adding different dtypes and converting back and forth
    event_array.features = pd.DataFrame(
        {"feature1": [1, 2, 3], "feature2": [4.0, 5.0, 6.0]}
    )
    remade_event_list = event_array.to_events([scan])
    assert len(remade_event_list) == 3
    remade_event_array = EventArray.from_events(remade_event_list)
    assert event_array == remade_event_array
    # Test saving and loading
    assert event_array.save_csv("tests/data/events.csv")
    assert event_array == EventArray.load_csv("tests/data/events.csv")
    os.remove("tests/data/events.csv")

    assert event_array.save_hdf5("tests/data/events.h5")
    assert event_array == EventArray.load_hdf5("tests/data/events.h5")
    os.remove("tests/data/events.h5")


def test_ocular_conversions():
    scan = Scan.load_txt("tests/data")
    input_path = "/mnt/HDSCA_Development/DZ/0B58703/ocular"
    result = EventArray.load_ocular(input_path)
    # For the purposes of this test, we will manually relabel "clust" == nan to 0
    # These come from ocular_interesting.rds, which does not have clusters
    result.metadata["clust"] = result.metadata["clust"].fillna(0)
    result.metadata["hcpc"] = result.metadata["hcpc"].fillna(0)
    result.save_ocular("tests/data")
    new_result = EventArray.load_ocular("tests/data")
    # # Sort them so that they are in the same order
    result = result.sort(["tile", "x", "y"])
    new_result = new_result.sort(["tile", "x", "y"])
    # Note: hcpc method within ocularr and here are different
    result.metadata["hcpc"] = new_result.metadata["hcpc"].copy()
    assert result == new_result
    # Clean up
    os.remove("tests/data/rc-final.csv")
    os.remove("tests/data/rc-final.rds")
    os.remove("tests/data/rc-final1.rds")
    os.remove("tests/data/rc-final2.rds")
    os.remove("tests/data/rc-final3.rds")
    os.remove("tests/data/rc-final4.rds")

    # Try it with "others" files
    result = EventArray.load_ocular(input_path, event_type="others")
    result.save_ocular("tests/data", event_type="others")
    new_result = EventArray.load_ocular("tests/data", event_type="others")
    result = result.sort(["tile", "x", "y"])
    new_result = new_result.sort(["tile", "x", "y"])
    # Note: hcpc method within ocularr and here are different
    result.metadata["hcpc"] = new_result.metadata["hcpc"].copy()
    assert result == new_result
    # Clean up
    os.remove("tests/data/others-final.csv")
    os.remove("tests/data/others-final.rds")
    os.remove("tests/data/others-final1.rds")
    os.remove("tests/data/others-final2.rds")
    os.remove("tests/data/others-final3.rds")
    os.remove("tests/data/others-final4.rds")


def test_copy_sort_rows_get():
    scan = Scan.load_yaml("tests/data")
    # Origin
    tile = Tile(scan, 0)
    events = [
        Event(scan, tile, 0, 100),
        Event(scan, tile, 0, 0),
        Event(scan, tile, 1000, 1000),
        Event(scan, tile, 1000, 1),
        Event(scan, tile, 2000, 2000),
    ]

    events = EventArray.from_events(events)

    # Copy
    events_copy = events.copy()
    events_copy.info["x"] = np.uint16(1)
    # Check that changes to the copy did not change the original
    assert events_copy.info["x"].equals(pd.Series([1, 1, 1, 1, 1], dtype=np.uint16))
    assert events.info["x"].equals(pd.Series([0, 0, 1000, 1000, 2000], dtype=np.uint16))

    # Sort
    events = events.sort(["x", "y"], ascending=[False, True])
    assert events.info["x"].equals(pd.Series([2000, 1000, 1000, 0, 0], dtype=np.uint16))
    assert events.info["y"].equals(pd.Series([2000, 1, 1000, 0, 100], dtype=np.uint16))

    # Get
    events_get = events.get(["x", "y"])
    assert events_get["x"].equals(pd.Series([2000, 1000, 1000, 0, 0], dtype=np.uint16))
    assert events_get["y"].equals(pd.Series([2000, 1, 1000, 0, 100], dtype=np.uint16))
    assert events_get.columns.equals(pd.Index(["x", "y"]))

    # Rows
    events_get = events.rows([0, 1, 3])
    assert len(events_get) == 3
    assert events_get.info["x"].equals(pd.Series([2000, 1000, 0], dtype=np.uint16))
    assert events_get.info["y"].equals(pd.Series([2000, 1, 0], dtype=np.uint16))
    events_get = events.rows([True, False, False, True, True])
    assert len(events_get) == 3
    assert events_get.info["x"].equals(pd.Series([2000, 0, 0], dtype=np.uint16))
    assert events_get.info["y"].equals(pd.Series([2000, 0, 100], dtype=np.uint16))


def test_event_montages():
    scan = Scan.load_txt("tests/data")
    tile = Tile(scan, 1000)
    event = Event(scan, tile, 515, 411)
    images = event.get_crops(crop_size=100)

    montage = csi_images.make_montage(
        images,
        [0, 1, 3, 2],
        {0: (0, 0, 1), 1: (1, 0, 0), 2: (0, 1, 0), 3: (1, 1, 1)},
        labels=["RGB", "DAPI", "AF555", "AF488", "AF647"],
    )
    if SHOW_PLOTS:
        cv2.imshow(
            "Full, classic montage with labels",
            cv2.cvtColor(montage, cv2.COLOR_RGB2BGR),
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def test_saving_crops_and_montages():
    scan = Scan.load_txt("tests/data")
    tile = Tile(scan, 1000)
    tile2 = Tile(scan, 0)
    events = [
        Event(scan, tile, 515, 411),
        Event(scan, tile2, 2, 1000),
        Event(scan, tile, 1000, 1000),
        Event(scan, tile2, 800, 800),
        Event(scan, tile, 1000, 662),
    ]

    # Get all crops and montages
    serial_crops = []
    serial_montages = []
    for event in events:
        serial_crops.append(event.get_crops())
        serial_montages.append(event.get_montage())

    # Save crops and montages
    Event.get_and_save_many_crops(events, "temp", scan.get_channel_names())
    Event.get_and_save_many_montages(events, "temp")

    saved_crops = []
    saved_montages = []
    for event in events:
        crops = event.load_crops("temp")
        saved_crops.append([crops[c] for c in scan.get_channel_names()])
        saved_montages.append(event.load_montage("temp"))

    # Make sure crops are identical
    for a, b in zip(serial_crops, saved_crops):
        for a_img, b_img in zip(a, b):
            assert np.array_equal(a_img, b_img)

    # Montages got JPEG compressed, so
    # Size comparison:
    for a, b in zip(serial_montages, saved_montages):
        assert a.shape == b.shape

    # Visual inspection
    if SHOW_PLOTS:
        cv2.imshow("Original", cv2.cvtColor(serial_montages[0], cv2.COLOR_RGB2BGR))
        cv2.imshow("Saved", cv2.cvtColor(saved_montages[0], cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Clean up
    for file in os.listdir("temp"):
        os.remove(os.path.join("temp", file))
    os.rmdir("temp")
