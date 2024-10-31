from csi_images import csi_tiles, csi_scans


def test_axioscan_tiles():
    scan = csi_scans.Scan.load_yaml("tests/data")
    tile_rows = scan.roi[0].tile_rows
    tile_cols = scan.roi[0].tile_cols
    tile = csi_tiles.Tile(scan, 0)
    assert tile.x == 0
    assert tile.y == 0

    tile = csi_tiles.Tile(scan, tile_cols - 1)
    assert tile.x == tile_cols - 1
    assert tile.y == 0

    tile = csi_tiles.Tile(scan, tile_cols)
    assert tile.x == 0
    assert tile.y == 1

    tile = csi_tiles.Tile(scan, tile_rows * tile_cols - 1)
    assert tile.x == tile_cols - 1
    assert tile.y == tile_rows - 1
    pass


def test_bzscanner_tiles():
    scan = csi_scans.Scan.load_txt("tests/data")
    tile_rows = scan.roi[0].tile_rows
    tile_cols = scan.roi[0].tile_cols
    tile = csi_tiles.Tile(scan, 0)
    assert tile.x == 0
    assert tile.y == 0

    tile = csi_tiles.Tile(scan, tile_cols - 1)
    assert tile.x == tile_cols - 1
    assert tile.y == 0

    tile = csi_tiles.Tile(scan, tile_cols)
    assert tile.x == tile_cols - 1
    assert tile.y == 1

    tile = csi_tiles.Tile(scan, 2 * tile_cols)
    assert tile.x == 0
    assert tile.y == 2
    pass


def test_getting_tiles():
    # Test getting all of the tiles
    scan = csi_scans.Scan.load_yaml("tests/data")
    tiles = csi_tiles.Tile.get_tiles(scan)
    assert len(tiles) == scan.roi[0].tile_rows * scan.roi[0].tile_cols
    tiles = csi_tiles.Tile.get_tiles(scan, as_flat=False)
    assert len(tiles) == scan.roi[0].tile_rows
    assert len(tiles[0]) == scan.roi[0].tile_cols

    # Just the first row
    tiles = csi_tiles.Tile.get_tiles_by_row_col(scan, rows=[0])
    assert len(tiles) == scan.roi[0].tile_cols
    assert all(tile.y == 0 for tile in tiles)
    assert all(tile.x == i for i, tile in enumerate(tiles))

    # Just the first column
    tiles = csi_tiles.Tile.get_tiles_by_row_col(scan, cols=[0])
    assert len(tiles) == scan.roi[0].tile_rows
    assert all(tile.x == 0 for tile in tiles)
    assert all(tile.y == i for i, tile in enumerate(tiles))

    # The bottom-right corner, with 4 tiles total
    tiles = csi_tiles.Tile.get_tiles_by_xy_bounds(
        scan,
        (
            scan.roi[0].tile_cols - 2,
            scan.roi[0].tile_rows - 2,
            scan.roi[0].tile_cols,
            scan.roi[0].tile_rows,
        ),
    )
    assert len(tiles) == 4
    assert tiles[0].x == scan.roi[0].tile_cols - 2
    assert tiles[0].y == scan.roi[0].tile_rows - 2
    assert tiles[1].x == scan.roi[0].tile_cols - 1
    assert tiles[1].y == scan.roi[0].tile_rows - 2
    assert tiles[2].x == scan.roi[0].tile_cols - 2
    assert tiles[2].y == scan.roi[0].tile_rows - 1
    assert tiles[3].x == scan.roi[0].tile_cols - 1
    assert tiles[3].y == scan.roi[0].tile_rows - 1
