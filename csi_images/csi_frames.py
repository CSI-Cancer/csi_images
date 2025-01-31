"""
Contains the Frame class, which represents a single frame of an image. The Frame class
does not hold the image data, but allows for easy loading of the image data from the
appropriate file. This module also contains functions for creating RGB and RGBW
composite images from a tile and a set of channels.
"""

import os
from typing import Self, Iterable

import numpy as np

from .csi_scans import Scan
from .csi_tiles import Tile

# Optional dependencies; will raise errors in particular functions if not installed
try:
    from . import csi_images
except ImportError:
    csi_images = None
try:
    import imageio.v3 as imageio
except ImportError:
    imageio = None


class Frame:
    def __init__(self, tile: Tile, channel: int | str):
        self.tile = tile
        if isinstance(channel, int):
            self.channel = channel
            if self.channel < 0 or self.channel >= len(tile.scan.channels):
                raise ValueError(
                    f"Channel index {self.channel} is out of bounds for scan."
                )
        elif isinstance(channel, str):
            self.channel = tile.scan.get_channel_indices([channel])[0]
        else:
            raise ValueError("Channel must be an integer or a string.")

    def __key(self) -> tuple:
        return self.tile, self.channel

    def __hash__(self) -> int:
        return hash(self.__key())

    def __repr__(self) -> str:
        return f"{self.tile}-{self.tile.scan.channels[self.channel].name}"

    def __eq__(self, other) -> bool:
        return self.__repr__() == other.__repr__()

    def get_file_path(self, input_path: str = None, ext: str = ".tif") -> str:
        """
        Get the file path for the frame, optionally changing
        the scan path and file extension.
        :param input_path: the path to the scan's directory. If None, defaults to
                           the path loaded in the frame's tile's scan object.
        :param ext: the image file extension. Defaults to .tif.
        :return: the file path.
        """
        if input_path is None:
            input_path = self.tile.scan.path
            if len(self.tile.scan.roi) > 1:
                input_path = os.path.join(input_path, f"roi_{self.tile.n_roi}")
        # Remove trailing slashes
        if input_path[-1] == os.sep:
            input_path = input_path[:-1]
        # Append proc if it's pointing to the base bzScanner directory
        if input_path.endswith("bzScanner"):
            input_path = os.path.join(input_path, "proc")
        # Should be a directory; append the file name
        if os.path.isdir(input_path):
            input_path = os.path.join(input_path, self.get_file_name(ext))
        else:
            raise ValueError(f"Input path {input_path} is not a directory.")
        return input_path

    def get_file_name(self, ext: str = ".tif") -> str:
        """
        Get the file name for the frame, handling different name conventions by scanner.
        :param ext: the image file extension. Defaults to .tif.
        :return: the file name.
        """
        if self.tile.scan.scanner_id.startswith(Scan.Type.AXIOSCAN7.value):
            channel_name = self.tile.scan.channels[self.channel].name
            x = self.tile.x
            y = self.tile.y
            file_name = f"{channel_name}-X{x:03}-Y{y:03}{ext}"
        elif self.tile.scan.scanner_id.startswith(Scan.Type.BZSCANNER.value):
            # BZScanner has channels in a specific order
            channel_name = self.tile.scan.channels[self.channel].name
            real_channel_index = list(
                self.tile.scan.BZSCANNER_CHANNEL_MAP.values()
            ).index(channel_name)
            # Determine total tiles
            roi = self.tile.scan.roi[self.tile.n_roi]
            total_tiles = roi.tile_rows * roi.tile_cols
            # Offset is based on total tiles and "real" channel index
            tile_offset = (real_channel_index * total_tiles) + 1  # 1-indexed
            n_bzscanner = self.tile.n + tile_offset
            file_name = f"Tile{n_bzscanner:06}{ext}"
        else:
            raise ValueError(f"Scanner {self.tile.scan.scanner_id} not supported.")
        return file_name

    def get_image(self, input_path: str = None, apply_gain: bool = True) -> np.ndarray:
        """
        Loads the image for this frame. Handles .tif (will return 16-bit images) and
        .jpg/.jpeg (will return 8-bit images), based on the CSI convention for storing
        .jpg/.jpeg images (compressed, using .tags files).
        :param input_path: the path to the scan's directory. If None, defaults to
                           the path loaded in the frame's tile's scan object.
        :param apply_gain: whether to apply the gain to the image. Only has an effect
                           if the scanner calculated but did not apply gain. Defaults to True.
        :return: the array representing the image.
        """
        file_path = self.get_file_path(input_path)

        # Check for the file
        if not os.path.exists(file_path):
            # Alternative: could be a .jpg/.jpeg file, test both
            if os.path.exists(os.path.splitext(file_path)[0] + ".jpg"):
                image = self._get_jpeg_image(os.path.splitext(file_path)[0] + ".jpg")
            elif os.path.exists(os.path.splitext(file_path)[0] + ".jpeg"):
                image = self._get_jpeg_image(os.path.splitext(file_path)[0] + ".jpeg")
            else:
                raise FileNotFoundError(
                    f"Could not find image at {file_path} or "
                    f"any format alternatives (.jpg, .jpeg)."
                )
        else:
            # Load the image
            if imageio is None:
                raise ModuleNotFoundError(
                    "imageio libraries not installed! "
                    "run `pip install csi_images[imageio]` to resolve."
                )
            image = imageio.imread(file_path)
        if image is None or image.size == 0:
            raise ValueError(f"Could not load image from {file_path}")
        if apply_gain and not self.tile.scan.channels[self.channel].gain_applied:
            # Multiply by the gain if it hasn't been applied, avoiding overflow
            dtype = image.dtype
            image = image.astype(np.uint32)
            image = image * self.tile.scan.channels[self.channel].intensity
            image = np.clip(image, 0, np.iinfo(dtype).max).astype(dtype)
        return image

    @staticmethod
    def _get_jpeg_image(input_path: str) -> np.ndarray:
        if imageio is None:
            raise ModuleNotFoundError(
                "imageio libraries not installed! "
                "run `pip install csi_images[imageio]` to resolve."
            )
        image = imageio.imread(input_path)
        if os.path.isfile(os.path.splitext(input_path)[0] + ".tags"):
            min_name = "PreservedMinValue"
            max_name = "PreservedMaxValue"
            min_value, max_value = -1, -1
            with open(os.path.splitext(input_path)[0] + ".tags", "r") as f:
                for line in f:
                    if line.startswith(min_name):
                        min_value = float(line.split("=")[1].strip())
                    elif line.startswith(max_name):
                        max_value = float(line.split("=")[1].strip())
                    if min_value != -1 and max_value != -1:
                        break
            if min_value != -1 and max_value != -1:
                if max_value > 1:
                    raise ValueError(f"{max_name} is greater than 1; unexpected .tags")
                if min_value < 0:
                    raise ValueError(f"{min_name} is less than 0; unexpected .tags")
                # Return to [0, 1], scale + offset, then return to a 16-bit image
                image = image / np.iinfo(image.dtype).max
                image = image * (max_value - min_value) + min_value
                image = (image * np.iinfo(np.uint16).max).astype(np.uint16)
            else:
                raise ValueError(
                    f"Could not find {min_name} and {max_name} in .tags file."
                )
        else:
            raise FileNotFoundError(f"Could not find .tags file for {input_path}")
        return image

    def check_image(self, input_path: str = None) -> bool:
        """
        Check if the image for this frame exists.
        :param input_path: the path to the scan's directory. If None, defaults to
                           the path loaded in the frame's tile's scan object.
        :return: whether the image exists.
        """
        file_path = self.get_file_path(input_path)
        # 72 is the minimum size for a valid TIFF file
        if os.path.exists(file_path) and os.path.getsize(file_path) > 72:
            return True
        else:
            # Alternative: could be a .jpg/.jpeg file, test both
            jpeg_path = os.path.splitext(file_path)[0] + ".jpg"
            if os.path.exists(jpeg_path) and os.path.getsize(jpeg_path) > 107:
                file_path = jpeg_path
            jpeg_path = os.path.splitext(file_path)[0] + ".jpeg"
            if os.path.exists(jpeg_path) and os.path.getsize(jpeg_path) > 107:
                file_path = jpeg_path
            # If we've found a .jpg/.jpeg, it must have a .tags file with it
            if file_path == jpeg_path:
                tags_path = os.path.splitext(file_path)[0] + ".tags"
                # Tags are text files that should include at least a few bytes
                if os.path.exists(tags_path) and os.path.getsize(tags_path) > 20:
                    return True
        # Didn't hit any of those, return false
        return False

    @classmethod
    def check_all_images(cls, scan: Scan):
        """
        Check if all images for a scan exist, either in .tif or .jpg form.
        :param scan:
        :return:
        """
        for n in range(len(scan.roi)):
            for frames in cls.get_all_frames(scan, n_roi=n):
                for frame in frames:
                    if not frame.check_image():
                        return False
        return True

    @classmethod
    def get_frames(
        cls,
        tile: Tile,
        channels: Iterable[int | str] = None,
    ) -> list[Self]:
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
            frames.append(Frame(tile, channel))
        return frames

    @classmethod
    def get_all_frames(
        cls,
        scan: Scan,
        channels: Iterable[int | str] = None,
        n_roi: int = 0,
        as_flat: bool = True,
    ) -> list[list[Self]] | list[list[list[Self]]]:
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
                tile = Tile(scan, n, n_roi)
                frames.append(cls.get_frames(tile, channels))
        else:
            frames = [[None] * scan.roi[n_roi].tile_cols] * scan.roi[n_roi].tile_rows
            for x in range(scan.roi[n_roi].tile_cols):
                for y in range(scan.roi[n_roi].tile_rows):
                    tile = Tile(scan, (x, y), n_roi)
                    frames[y][x] = cls.get_frames(tile, channels)
        return frames

    @classmethod
    def make_rgb_image(
        cls,
        tile: Tile,
        channels: dict[int, tuple[float, float, float]],
        input_path=None,
    ) -> np.ndarray:
        """
        Convenience method for creating an RGB image from a tile and a set of channels
        without manually extracting any frames.
        :param tile: the tile for which the image should be made.
        :param channels: a dictionary of scan channel indices and RGB gains.
        :param input_path: the path to the input images. Will use metadata if not provided.
        :return: the image as a numpy array.
        """
        if csi_images is None:
            raise ModuleNotFoundError(
                "imageio libraries not installed! "
                "run `pip install csi_images[imageio]` to resolve."
            )
        images = []
        colors = []
        for channel_index, color in channels.items():
            if channel_index == -1:
                continue
            image = Frame(tile, channel_index).get_image(input_path)
            images.append(image)
            colors.append(color)
        return csi_images.make_rgb(images, colors)
