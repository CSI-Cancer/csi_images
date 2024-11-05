"""
Contains the Frame class, which represents a single frame of an image. The Frame class
does not hold the image data, but allows for easy loading of the image data from the
appropriate file. This module also contains functions for creating RGB and RGBW
composite images from a tile and a set of channels.
"""

import os
import typing

import cv2
import numpy as np

from csi_images import csi_scans, csi_tiles


class Frame:
    def __init__(self, scan: csi_scans.Scan, tile: csi_tiles.Tile, channel: int | str):
        self.scan = scan
        self.tile = tile
        if isinstance(channel, int):
            self.channel = channel
            if self.channel < 0 or self.channel >= len(scan.channels):
                raise ValueError(
                    f"Channel index {self.channel} is out of bounds for scan."
                )
        elif isinstance(channel, str):
            self.channel = self.scan.get_channel_indices([channel])

    def __repr__(self) -> str:
        return f"{self.scan.slide_id}-{self.tile.n}-{self.scan.channels[self.channel].name}"

    def __eq__(self, other) -> bool:
        return self.__repr__() == other.__repr__()

    def get_file_name(self, file_extension: str = ".tif") -> str:
        if self.scan.scanner_id.startswith(csi_scans.Scan.Type.AXIOSCAN7.value):
            channel_name = self.scan.channels[self.channel].name
            x = self.tile.x
            y = self.tile.y
            file_name = f"{channel_name}-X{x:03}-Y{y:03}{file_extension}"
        elif self.scan.scanner_id.startswith(csi_scans.Scan.Type.BZSCANNER.value):
            channel_name = self.scan.channels[self.channel].name
            real_channel_index = list(self.scan.BZSCANNER_CHANNEL_MAP.values()).index(
                channel_name
            )
            total_tiles = self.scan.roi[0].tile_rows * self.scan.roi[0].tile_cols
            tile_offset = (real_channel_index * total_tiles) + 1  # 1-indexed
            n_bzscanner = self.tile.n + tile_offset
            file_name = f"Tile{n_bzscanner:06}{file_extension}"
        else:
            raise ValueError(f"Scanner {self.scan.scanner_id} not supported.")
        return file_name

    def get_image(self, input_path: str = None) -> tuple[np.ndarray, str]:
        """
        Load the image from the appropriate file, automatically determining the path.
        :param input_path: an optional, manual input path. Can be a directory
        (will append the file name) or a file.
        :return: the array representing the image and the path to the file.
        """
        if input_path is None:
            input_path = self.scan.path
            if len(self.scan.roi) > 1:
                input_path = os.path.join(input_path, f"roi_{self.tile.n_roi}")
        # Remove trailing slashes
        if input_path[-1] == os.sep:
            input_path = input_path[:-1]
        # Append proc if it's pointing to the base bzScanner directory
        if input_path.endswith("bzScanner"):
            input_path = os.path.join(input_path, "proc")
        if os.path.isdir(input_path):
            input_path = os.path.join(input_path, self.get_file_name())

        # Check for the file
        if not os.path.exists(input_path):
            # Alternative: it's a .jpg/.jpeg file
            temporary_path = os.path.splitext(input_path)[0] + ".jpg"
            if os.path.exists(temporary_path):
                input_path = temporary_path
            temporary_path = os.path.splitext(input_path)[0] + ".jpeg"
            if os.path.exists(temporary_path):
                input_path = temporary_path
            # If we've found a .jpg/.jpeg, try loading it as compressed
            if input_path == temporary_path:
                return self._get_jpeg_image(input_path), input_path
            else:
                raise FileNotFoundError(f"Could not find file {input_path}")

        image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if image is None or image.size == 0:
            raise ValueError(f"Could not load image from {input_path}")
        return image, input_path

    def _get_jpeg_image(self, input_path: str) -> np.ndarray:
        raise NotImplementedError("JPEG image loading not yet implemented.")

    @classmethod
    def get_frames(
        cls, tile: csi_tiles.Tile, channels: tuple[int | str] = None
    ) -> list[typing.Self]:
        """
        Get the frames for a tile and a set of channels. By default, gets all channels.
        :param tile: the tile.
        :param channels: the channels, as indices or names. Defaults to all channels.
        :return: the frames, in order of the channels.
        """
        if channels is None:
            channels = range(len(tile.scan.channels))
        frames = []
        for channel in channels:
            frames.append(Frame(tile.scan, tile, channel))
        return frames

    @classmethod
    def get_all_frames(
        cls,
        scan: csi_scans.Scan,
        channels: tuple[int | str] = None,
        n_roi: int = 0,
        as_flat: bool = True,
    ) -> list[list[typing.Self]] | list[list[list[typing.Self]]]:
        """
        Get all frames for a scan and a set of channels.
        :param scan: the scan metadata.
        :param channels: the channels, as indices or names. Defaults to all channels.
        :param n_roi: the region of interest to use. Defaults to 0.
        :param as_flat: whether to flatten the frames into a 2D list.
        :return: if as_flat: 2D list of frames, organized as [n][channel];
                 if not as_flat: 3D list of frames organized as [row][col][channel] a.k.a. [y][x][channel].
        """
        if as_flat:
            frames = []
            for n in range(scan.roi[n_roi].tile_rows * scan.roi[n_roi].tile_cols):
                tile = csi_tiles.Tile(scan, n, n_roi)
                frames.append(cls.get_frames(tile, channels))
        else:
            frames = [[None] * scan.roi[n_roi].tile_cols] * scan.roi[n_roi].tile_rows
            for x in range(scan.roi[n_roi].tile_cols):
                for y in range(scan.roi[n_roi].tile_rows):
                    tile = csi_tiles.Tile(scan, (x, y), n_roi)
                    frames[y][x] = cls.get_frames(tile, channels)
        return frames


def make_rgb_image(
    tile: csi_tiles.Tile,
    channel_indices: tuple[int, int, int],
    channel_gains: tuple[float, float, float] = (1.0, 1.0, 1.0),
    input_path=None,
):
    """

    :param tile: the tile for which the image should be made.
    :param channel_indices: the indices of the channels to use, in order of RGB.
    :param channel_gains: the gains for each channel, in order of RGB.
    :param input_path: the path to the input images. Will use metadata if not provided.
    :return: the image as a numpy array.
    """
    if len(channel_indices) != 3 or len(channel_gains) != 3:
        raise ValueError("Channel indices and gains must have 3 elements.")

    red_index, green_index, blue_index = channel_indices
    red_gain, green_gain, blue_gain = channel_gains
    height, width = tile.scan.tile_height_px, tile.scan.tile_width_px
    # Load images for each channel, making a blank image if the channel is -1
    if red_index == -1:
        red_image = np.zeros((height, width))
    else:
        red_image = Frame(tile.scan, tile, red_index).get_image(input_path)[0]
    if green_index == -1:
        green_image = np.zeros((height, width))
    else:
        green_image = Frame(tile.scan, tile, green_index).get_image(input_path)[0]
    if blue_index == -1:
        blue_image = np.zeros((height, width))
    else:
        blue_image = Frame(tile.scan, tile, blue_index).get_image(input_path)[0]

    # Create output matrix, larger than needed to avoid overflow
    output = np.zeros((height, width, 3)).astype(np.uint32)
    output[:, :, 0] = red_image * red_gain
    output[:, :, 1] = green_image * green_gain
    output[:, :, 2] = blue_image * blue_gain
    # Cap it off at 65535
    output = np.clip(output, 0, 65535).astype(np.uint16)
    return output


def make_rgbw_image(
    tile: csi_tiles.Tile,
    channel_indices: tuple[int, int, int, int],
    channel_gains: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    input_path=None,
):
    """

    :param tile: the tile for which the image should be made.
    :param channel_indices: the indices of the channels to use, in order of RGBW.
    :param channel_gains: the gains for each channel, in order of RGBW.
    :param input_path: the path to the input images. Will use metadata if not provided.
    :return: the image as a numpy array.
    """
    if len(channel_indices) != 4 or len(channel_gains) != 4:
        raise ValueError("Channel indices and gains must have 4 elements.")

    red_index, green_index, blue_index, white_index = channel_indices
    red_gain, green_gain, blue_gain, white_gain = channel_gains
    height, width = tile.scan.tile_height_px, tile.scan.tile_width_px
    # Load images for each channel, making a blank image if the channel is -1
    if red_index == -1:
        red_image = np.zeros((height, width))
    else:
        red_image = Frame(tile.scan, tile, red_index).get_image(input_path)[0]
    if green_index == -1:
        green_image = np.zeros((height, width))
    else:
        green_image = Frame(tile.scan, tile, green_index).get_image(input_path)[0]
    if blue_index == -1:
        blue_image = np.zeros((height, width))
    else:
        blue_image = Frame(tile.scan, tile, blue_index).get_image(input_path)[0]
    if white_index == -1:
        white_image = np.zeros((height, width))
    else:
        white_image = Frame(tile.scan, tile, white_index).get_image(input_path)[0]

    # Create output matrix, larger than needed to avoid overflow
    output = np.zeros((height, width, 3)).astype(np.uint32)
    output[:, :, 0] = red_image * red_gain + white_image * white_gain
    output[:, :, 1] = green_image * green_gain + white_image * white_gain
    output[:, :, 2] = blue_image * blue_gain + white_image * white_gain
    # Cap it off at 65535
    output = np.clip(output, 0, 65535).astype(np.uint16)
    return output
