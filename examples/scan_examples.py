#!/usr/bin/env python

import os
from csi_images import csi_scans, csi_tiles, csi_frames, csi_events


# Create a basic DatabaseHandler
def load_in_images():
    repository_path = os.path.dirname(os.path.dirname(__file__))
    test_data_path = os.path.join(repository_path, "tests", "data")

    # First, let's load in a scan's metadata
    scan = csi_scans.Scan.load_yaml(test_data_path)

    # Using that metadata, we can load in a tile or a ton of tiles
    tile = csi_tiles.Tile(scan, 0)
    tiles = csi_tiles.Tile.get_tiles(scan)

    # By default, these will load in a single list
    assert len(tiles) == scan.roi[0].tile_rows * scan.roi[0].tile_cols

    # But we can also load them in as a grid
    tiles = csi_tiles.Tile.get_tiles(scan, as_flat=False)
    assert len(tiles) == scan.roi[0].tile_rows
    assert len(tiles[0]) == scan.roi[0].tile_cols

    # We can also load in frames for a tile or a ton of tiles
    frames = csi_frames.Frame.get_frames(tile)
    all_frames = csi_frames.Frame.get_all_frames(scan)
    assert len(all_frames) == scan.roi[0].tile_rows * scan.roi[0].tile_cols
    assert len(all_frames[0]) == 4
    all_frames = csi_frames.Frame.get_all_frames(scan, as_flat=False)
    assert len(all_frames) == scan.roi[0].tile_rows
    assert len(all_frames[0]) == scan.roi[0].tile_cols
    assert len(all_frames[0][0]) == 4

    # And for each frame, we can load the actual image
    # First element is the image array, second element is the file path it came from
    image = frames[0].get_image()
    assert image.shape == (scan.tile_height_px, scan.tile_width_px)


if __name__ == "__main__":
    load_in_images()
