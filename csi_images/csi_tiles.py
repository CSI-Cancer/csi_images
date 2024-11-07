"""
Contains the Tile class, which represents a collection of frames at the same position
in a scan. The module comes with several helper functions that allow for gathering tiles
based on their position in the scan.
"""

import typing

import numpy as np

from csi_images.csi_scans import Scan


class Tile:
    """
    A class that represents a tile in a scan. This class encodes the position of a group
    of frames in a scan, based on the scan's metadata. The module comes with several
    helper functions that allow for gathering tiles based on their position in the scan.
    """

    def __init__(self, scan: Scan, coordinates: int | tuple[int, int], n_roi: int = 0):
        self.scan = scan
        tile_rows = scan.roi[n_roi].tile_rows
        tile_cols = scan.roi[n_roi].tile_cols
        total_tiles = tile_rows * tile_cols
        # Check that the n_roi is valid
        if n_roi >= len(self.scan.roi):
            raise ValueError(f"n_roi {n_roi} is out of bounds for scan.")
        self.n_roi = int(n_roi)
        if np.issubdtype(type(coordinates), np.integer):
            # We received "n" as the coordinates
            if 0 > coordinates or coordinates > total_tiles:
                raise ValueError(
                    f"n ({coordinates}) must be between 0 and the "
                    f"number of tiles in ROI {self.n_roi} ({total_tiles})."
                )
            self.n = int(coordinates)
            self.x, self.y = self.n_to_position()
        elif (
            (isinstance(coordinates, tuple) or isinstance(coordinates, list))
            and len(coordinates) == 2
            and all([np.issubdtype(type(coord), np.integer) for coord in coordinates])
        ):
            # We received (x, y) as the coordinates
            if 0 > coordinates[0] or coordinates[0] >= tile_cols:
                raise ValueError(
                    f"x ({coordinates[0]}) must be between 0 and the "
                    f"number of columns in ROI {self.n_roi} ({tile_cols})."
                )
            if 0 > coordinates[1] or coordinates[1] >= tile_rows:
                raise ValueError(
                    f"y ({coordinates[1]}) must be between 0 and the "
                    f"number of rows in ROI {self.n_roi} ({tile_rows})."
                )
            self.x, self.y = int(coordinates[0]), int(coordinates[1])
            self.n = self.position_to_n()
        else:
            raise ValueError(
                "Coordinates must be an integer n or a tuple of (x, y) coordinates."
            )

    def __repr__(self) -> str:
        return f"{self.scan.slide_id}-{self.n}"

    def __eq__(self, other) -> bool:
        return self.__repr__() == other.__repr__()

    # Helper functions that convert ***indices***, which are 0-indexed
    def position_to_n(self, position: tuple[int, int] = (-1, -1)) -> int:
        """
        Convert the x, y coordinates to the n coordinate, based on this tile's scan
        metadata and ROI. Can be provided alternative x, y to convert for convenience.
        :param position: optional (x, y) coordinates to find the n for.
                         If none provided, this tile's (x, y) will be used.
        :return: the coordinate n, which depends on the scanner and scan layout.
        """
        if position == (-1, -1):
            position = self.x, self.y
        x, y = position
        if self.scan.scanner_id.startswith(self.scan.Type.AXIOSCAN7.value):
            n = y * self.scan.roi[self.n_roi].tile_cols + x
        elif self.scan.scanner_id.startswith(self.scan.Type.BZSCANNER.value):
            n = y * self.scan.roi[self.n_roi].tile_cols
            if y % 2 == 0:
                n += x
            else:
                n += self.scan.roi[0].tile_cols - x
        else:
            raise ValueError(f"Scanner type {self.scan.scanner_id} not supported.")
        return n

    def n_to_position(self, n: int = -1) -> tuple[int, int]:
        """
        Convert the n coordinate to x, y coordinates, based on this tile's scan
        metadata and ROI. Can be provided alternative n to convert for convenience.
        :param n: an optional n coordinate to find the position for.
                  If none provided, this tile's n will be used.
        :return: x, y coordinates of the tile in the scan's coordinate system.
        """
        if n == -1:
            n = self.n
        if n < 0:
            raise ValueError(f"n ({n}) must be non-negative.")
        if self.scan.scanner_id.startswith(self.scan.Type.AXIOSCAN7.value):
            x = n % self.scan.roi[0].tile_cols
            y = n // self.scan.roi[0].tile_cols
            return x, y
        elif self.scan.scanner_id.startswith(self.scan.Type.BZSCANNER.value):
            y = n // self.scan.roi[0].tile_cols
            if y % 2 == 0:
                x = n % self.scan.roi[0].tile_cols
            else:
                x = (self.scan.roi[0].tile_cols - 1) - (n % self.scan.roi[0].tile_cols)
        else:
            raise ValueError(f"Scanner type {self.scan.scanner_id} not supported.")
        return x, y

    @classmethod
    def get_tiles(
        cls,
        scan: Scan,
        coordinates: list[int] | list[tuple[int, int]] = None,
        n_roi: int = 0,
        as_flat: bool = True,
    ) -> list[typing.Self] | list[list[typing.Self]]:
        """
        The simplest way to gather a list of Tile objects. By default, it will gather all
        tiles in the scan. To gather specific tiles, provide a list of coordinates.
        :param scan: the scan metadata.
        :param coordinates: a list of n-based indices or (x, y) coordinates.
                            Leave as None to include all tiles.
        :param n_roi: the region of interest to use. Defaults to 0.
        :param as_flat: whether to return a flat list of Tile objects or a list of lists.
        :return: if as_flat: a list of Tile objects in the same order as the coordinates;
                 if not as_flat: a list of lists of Tile objects in their relative coordinates.
        """
        if as_flat:
            if coordinates is None:
                # Populate coordinates with all n's.
                coordinates = list(
                    range(scan.roi[n_roi].tile_rows * scan.roi[n_roi].tile_cols)
                )
            tiles = []
            for coordinate in coordinates:
                tiles.append(cls(scan, coordinate, n_roi))
        else:
            if coordinates is None:
                # Populate coordinates with all (x, y) pairs in row-major order
                coordinates = []
                for y in range(scan.roi[n_roi].tile_rows):
                    for x in range(scan.roi[n_roi].tile_cols):
                        coordinates.append((x, y))
            # Check that the coordinates are contiguous, otherwise we can't make a grid
            # Find the min and max x, y values
            x_min = scan.roi[n_roi].tile_cols
            x_max = 0
            y_min = scan.roi[n_roi].tile_rows
            y_max = 0
            for x, y in coordinates:
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)

            # Check that the coordinates are contiguous
            if (x_max - x_min + 1) * (y_max - y_min + 1) != len(coordinates):
                raise ValueError(
                    "Coordinates must be a contiguous square to form "
                    "a grid; number of coordinates does not match."
                )

            tiles = [[None] * (x_max - x_min + 1)] * (y_max - y_min + 1)
            for coordinate in coordinates:
                x, y = coordinate
                tiles[y][x] = cls(scan, coordinate, n_roi)

        return tiles

    @classmethod
    def get_tiles_by_row_col(
        cls,
        scan: Scan,
        rows: list[int] = None,
        cols: list[int] = None,
        n_roi: int = 0,
        as_flat: bool = True,
    ) -> list[typing.Self] | list[list[typing.Self]]:
        """
        Gather a list of Tile objects based on the row and column indices provided.
        If left as None, it will gather all rows and/or columns.
        :param scan: the scan metadata.
        :param rows: a list of 0-indexed rows (y-positions) in the scan axes.
                     Leave as None to include all rows.
        :param cols: a list of 0-indexed columns (x-positions) in the scan axes.
                     Leave as None to include all columns.
        :param n_roi: the region of interest to use. Defaults to 0.
        :param as_flat: whether to return a flat list of Tile objects or a list of lists.
        :return: if as_flat: a list of Tile objects in row-major order;
                 if not as_flat: a list of lists of Tile objects in their relative coordinates
        """
        if rows is None:
            rows = list(range(scan.roi[n_roi].tile_rows))
        if cols is None:
            cols = list(range(scan.roi[n_roi].tile_cols))

        # Populate coordinates
        coordinates = []
        for row in rows:
            for col in cols:
                coordinates.append((col, row))

        return cls.get_tiles(scan, coordinates, n_roi, as_flat)

    @classmethod
    def get_tiles_by_xy_bounds(
        cls,
        scan: Scan,
        bounds: tuple[int, int, int, int],
        n_roi: int = 0,
        as_flat: bool = True,
    ) -> list[typing.Self] | list[list[typing.Self]]:
        """
        Gather a list of Tile objects based on the x, y bounds provided. The bounds are
        exclusive, like indices, so the tiles at the corners are NOT included in the list.
        :param scan: the scan metadata.
        :param bounds: a tuple of (x_0, y_0, x_1, y_1) in the scan axes.
        :param n_roi: the region of interest to use. Defaults to 0.
        :param as_flat: whether to return a flat list of Tile objects or a list of lists.
        :return: if as_flat: a list of Tile objects in row-major order;
                 if not as_flat: a list of lists of Tile objects in their relative coordinates
        """
        x_0, y_0, x_1, y_1 = bounds
        coordinates = []
        for y in range(y_0, y_1):
            for x in range(x_0, x_1):
                coordinates.append((x, y))
        return cls.get_tiles(scan, coordinates, n_roi, as_flat)
