#!/usr/bin/env python

import os
from csi_images.csi_scans import Scan
from csi_images.csi_tiles import Tile
from csi_images.csi_frames import Frame


# Create a basic DatabaseHandler
def load_in_images():
    repository_path = os.path.dirname(os.path.dirname(__file__))
    test_data_path = os.path.join(repository_path, "tests", "data")

    # First, let's load in a scan's metadata
    scan = Scan.load_yaml(test_data_path)

    # Using that metadata, we can load in a tile or a ton of tiles
    tile = Tile(scan, 0)
    tiles = Tile.get_tiles(scan)

    # By default, these will load in a single list
    assert len(tiles) == scan.roi[0].tile_rows * scan.roi[0].tile_cols

    # But we can also load them in as a grid
    tiles = Tile.get_tiles(scan, as_flat=False)
    assert len(tiles) == scan.roi[0].tile_rows
    assert len(tiles[0]) == scan.roi[0].tile_cols

    # We can also load in frames for a tile or a ton of tiles
    frames = Frame.get_frames(tile)
    all_frames = Frame.get_all_frames(scan)
    assert len(all_frames) == scan.roi[0].tile_rows * scan.roi[0].tile_cols
    assert len(all_frames[0]) == 4
    all_frames = Frame.get_all_frames(scan, as_flat=False)
    assert len(all_frames) == scan.roi[0].tile_rows
    assert len(all_frames[0]) == scan.roi[0].tile_cols
    assert len(all_frames[0][0]) == 4

    # And for each frame, we can load the actual image
    # First element is the image array, second element is the file path it came from
    image = frames[0].get_image()
    assert image.shape == scan.get_image_size()


if __name__ == "__main__":
    load_in_images()
