from csi_images.csi_scans import Scan
from csi_images.csi_tiles import Tile


def test_axioscan_tiles():
    scan = Scan.load_yaml("tests/data")
    tile = Tile(scan, 0)
    # Assert that the tile is in the top-left corner
    assert (tile.x, tile.y) == (0, 0)
    assert tile.position_to_n() == 0
    assert tile.n_to_position() == (0, 0)

    # Assert that position_to_n() and n_to_position() work with different positions
    assert tile.position_to_n((1, 0)) == 1
    assert tile.n_to_position(1) == (1, 0)

    assert tile.position_to_n((0, 1)) == scan.roi[0].tile_cols
    assert tile.n_to_position(scan.roi[0].tile_cols) == (0, 1)

    assert (
        tile.position_to_n((scan.roi[0].tile_cols - 1, scan.roi[0].tile_rows - 1))
        == scan.roi[0].tile_cols * scan.roi[0].tile_rows - 1
    )
    assert tile.n_to_position(scan.roi[0].tile_cols * scan.roi[0].tile_rows - 1) == (
        scan.roi[0].tile_cols - 1,
        scan.roi[0].tile_rows - 1,
    )

    # Assert that creating a tile with a position works correctly
    tile = Tile(scan, (1, 0))
    assert tile.n == 1

    # Assert that creating a tile with a bad n_roi raises an error
    try:
        Tile(scan, 0, 1)
        assert False
    except ValueError:
        assert True


def test_bzscanner_tiles():
    scan = Scan.load_txt("tests/data")
    # Origin
    tile = Tile(scan, 0)
    assert (tile.x, tile.y) == (0, 0)
    tile = Tile(scan, (0, 0))
    assert tile.n == 0
    # Assert row-major indexing
    tile = Tile(scan, 1)
    assert (tile.x, tile.y) == (1, 0)
    tile = Tile(scan, (1, 0))
    assert tile.n == 1
    # Assert snake indexing
    tile = Tile(scan, scan.roi[0].tile_cols - 1)
    assert (tile.x, tile.y) == (scan.roi[0].tile_cols - 1, 0)
    tile = Tile(scan, (scan.roi[0].tile_cols - 1, 0))
    assert tile.n == scan.roi[0].tile_cols - 1
    tile = Tile(scan, scan.roi[0].tile_cols)
    assert (tile.x, tile.y) == (scan.roi[0].tile_cols - 1, 1)
    tile = Tile(scan, (scan.roi[0].tile_cols - 1, 1))
    assert tile.n == scan.roi[0].tile_cols
    # Assert snake indexing again
    tile = Tile(scan, scan.roi[0].tile_cols * 2 - 1)
    assert (tile.x, tile.y) == (0, 1)
    tile = Tile(scan, (0, 1))
    assert tile.n == scan.roi[0].tile_cols * 2 - 1
    tile = Tile(scan, scan.roi[0].tile_cols * 2)
    assert (tile.x, tile.y) == (0, 2)
    tile = Tile(scan, (0, 2))
    assert tile.n == scan.roi[0].tile_cols * 2


def test_getting_tiles():
    scan = Scan.load_yaml("tests/data")
    # All tiles
    tiles = Tile.get_tiles(scan)
    assert len(tiles) == scan.roi[0].tile_rows * scan.roi[0].tile_cols
    # All tiles, as a grid
    tiles = Tile.get_tiles(scan, as_flat=False)
    assert len(tiles) == scan.roi[0].tile_rows
    assert len(tiles[0]) == scan.roi[0].tile_cols

    # Just the first row
    tiles = Tile.get_tiles_by_row_col(scan, rows=[0])
    assert len(tiles) == scan.roi[0].tile_cols
    assert all(tile.y == 0 for tile in tiles)
    assert all(tile.x == i for i, tile in enumerate(tiles))

    # Just the first column
    tiles = Tile.get_tiles_by_row_col(scan, cols=[0])
    assert len(tiles) == scan.roi[0].tile_rows
    assert all(tile.x == 0 for tile in tiles)
    assert all(tile.y == i for i, tile in enumerate(tiles))

    # The bottom-right corner, with 4 tiles total
    tiles = Tile.get_tiles_by_xy_bounds(
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
