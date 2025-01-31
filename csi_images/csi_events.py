"""
Contains the Event class, which represents a single event in a scan.
The Event class optionally holds metadata and features. Lists of events with
similar metadata or features can be combined into DataFrames for analysis.

The Event class holds the position of the event in the frame, which can be converted
to the position in the scanner or slide coordinate positions. See the
csi_utils.csi_scans documentation page for more information on the coordinate systems.
"""

import os
import glob
import math
import warnings
from typing import Self, Iterable, Hashable, Sequence

import numpy as np
import pandas as pd

from .csi_scans import Scan
from .csi_tiles import Tile
from .csi_frames import Frame

# Optional dependencies; will raise errors in particular functions if not installed
try:
    from . import csi_images
except ImportError:
    csi_images = None
try:
    import imageio.v3 as imageio
except ImportError:
    imageio = None
try:
    import pyreadr
except ImportError:
    pyreadr = None


class Event:
    """
    A class that represents a single event in a scan, making it easy to evaluate
    singular events. Required metadata is exposed as attributes, and optional
    metadata and features are stored as DataFrames.
    """

    SCAN_TO_SLIDE_TRANSFORM = {
        # Axioscan zero is in the top-right corner instead of top-left
        Scan.Type.AXIOSCAN7: np.array(
            [
                [1, 0, 75000],
                [0, 1, 0],
                [0, 0, 1],
            ]
        ),
        # BZScanner coordinates are a special kind of messed up:
        # - The slide is upside-down.
        # - The slide is oriented vertically, with the barcode at the bottom.
        # - Tiles are numbered from the top-right
        Scan.Type.BZSCANNER: np.array(
            [
                [0, -1, 75000],
                [-1, 0, 25000],
                [0, 0, 1],
            ]
        ),
    }
    """
    Homogeneous transformation matrices for converting between scanner and slide
    coordinates. The matrices are 3x3, with the final column representing the
    translation in micrometers (um). For more information, see 
    [affine transformations](https://en.wikipedia.org/wiki/Transformation_matrix#Affine_transformations).
    
    Transformations are nominal, and accuracy is not guaranteed; this is due to 
    imperfections in slides and alignment in the scanners. Units are in micrometers.
    """

    def __init__(
        self,
        tile: Tile,
        x: int,
        y: int,
        metadata: pd.Series = None,
        features: pd.Series = None,
    ):
        self.tile = tile
        self.x = int(x)
        self.y = int(y)
        self.metadata = metadata
        self.features = features

    def __repr__(self) -> str:
        return f"{self.tile}-{self.x}-{self.y}"

    def __eq__(self, other) -> bool:
        return self.__repr__() == other.__repr__()

    def __lt__(self, other):
        return self.__repr__() < other.__repr__()

    def get_scan_position(self) -> tuple[float, float]:
        """
        Get the position of the event in the scanner's coordinate frame.
        :return: the scan position of the event in micrometers (um).
        """
        # Get overall pixel position
        real_tile_height, real_tile_width = self.tile.scan.get_image_size()
        pixel_x = self.x + (real_tile_width * self.tile.x)
        pixel_y = self.y + (real_tile_height * self.tile.y)
        # Convert to micrometers
        x_um = pixel_x * self.tile.scan.pixel_size_um
        y_um = pixel_y * self.tile.scan.pixel_size_um
        # Add the scan's origin in the scanner frame
        x_um += self.tile.scan.roi[self.tile.n_roi].origin_x_um
        y_um += self.tile.scan.roi[self.tile.n_roi].origin_y_um
        return x_um, y_um

    def get_slide_position(self) -> tuple[float, float]:
        """
        Get the slide position of the event in micrometers (um).
        :return: the slide position of the event.
        """
        # Turn scan_position into a 3x1 vector
        scan_position = self.get_scan_position()
        scan_position = np.array([[scan_position[0]], [scan_position[1]], [1]])

        # Multiply by the appropriate homogeneous matrix
        if self.tile.scan.scanner_id.startswith(self.tile.scan.Type.AXIOSCAN7.value):
            transform = self.SCAN_TO_SLIDE_TRANSFORM[self.tile.scan.Type.AXIOSCAN7]
        elif self.tile.scan.scanner_id.startswith(self.tile.scan.Type.BZSCANNER.value):
            transform = self.SCAN_TO_SLIDE_TRANSFORM[self.tile.scan.Type.BZSCANNER]
        else:
            raise ValueError(f"Scanner type {self.tile.scan.scanner_id} not supported.")
        slide_position = np.matmul(transform, scan_position)
        return float(slide_position[0][0]), float(slide_position[1][0])

    def crop(
        self, images: Iterable[np.ndarray], crop_size: int = 100, in_pixels: bool = True
    ) -> list[np.ndarray]:
        """
        Crop the event from the provided frame images. Use if you have already gotten
        frame images; useful for cropping multiple events from the same frame image.
        :param images: the frame images.
        :param crop_size: the square size of the image crop to get for this event.
        :param in_pixels: whether the crop size is in pixels or micrometers. Defaults to pixels.
        :return: image_size x image_size crops of the event in the provided frames. If
        the event is too close to the edge, the crop will be smaller and not centered.
        """
        # Convert a crop size in micrometers to pixels
        if not in_pixels:
            crop_size = round(crop_size / self.tile.scan.pixel_size_um)
        image_height, image_width = 0, 0
        for image in images:
            if image_height == 0 and image_width == 0:
                image_height, image_width = image.shape
            else:
                if image_height != image.shape[0] or image_width != image.shape[1]:
                    raise ValueError("All images must be the same size")
        if image_height == 0 or image_width == 0:
            raise ValueError("No images provided")

        # Find the crop bounds
        bounds = [
            self.x - (crop_size // 2) + 1,
            self.y - (crop_size // 2) + 1,
            self.x + math.ceil(crop_size / 2) + 1,
            self.y + math.ceil(crop_size / 2) + 1,
        ]
        # Determine how much the bounds violate the image size
        displacements = [
            max(0, -bounds[0]),
            max(0, -bounds[1]),
            max(0, bounds[2] - image_width),
            max(0, bounds[3] - image_height),
        ]
        # Cap off the bounds
        bounds = [
            max(0, bounds[0]),
            max(0, bounds[1]),
            min(image_width, bounds[2]),
            min(image_height, bounds[3]),
        ]

        # Crop the images
        crops = []
        for image in images:
            # Create a blank image of the right size
            crop = np.zeros((crop_size, crop_size), dtype=image.dtype)

            # Insert the cropped image into the blank image, leaving a black buffer
            # around the edges if the crop would go beyond the original image bounds
            crop[
                displacements[1] : crop_size - displacements[3],
                displacements[0] : crop_size - displacements[2],
            ] = image[bounds[1] : bounds[3], bounds[0] : bounds[2]]
            crops.append(crop)
        return crops

    def get_crops(
        self,
        crop_size: int = 100,
        in_pixels: bool = True,
        input_path: str = None,
        channels: Iterable[int | str] = None,
        apply_gain: bool | Iterable[bool] = True,
    ) -> list[np.ndarray]:
        """
        Gets the frame images for this event and then crops the event from the images.
        Convenient for retrieving a single event's crops, but less efficient when
        retrieving multiple events from the same tile as it will reread the images.
        :param crop_size: the square size of the image crop to get for this event.
        :param in_pixels: whether the crop size is in pixels or micrometers. Defaults to pixels.
        :param input_path: the path to the input images. Defaults to None (uses the scan's path).
        :param channels: the channels to extract images for. Defaults to all channels.
        :param apply_gain: whether to apply scanner-calculated gain to the images, if
        not already applied. If a list, matches the channels.
        :return: a list of cropped images from the scan in the order of the channels.
        """
        # This function validates channels
        frames = Frame.get_frames(self.tile, channels)
        # Convert individual inputs to lists of appropriate length
        if isinstance(apply_gain, bool):
            apply_gain = [apply_gain] * len(frames)
        images = [f.get_image(input_path, a) for f, a in zip(frames, apply_gain)]
        return self.crop(images, crop_size, in_pixels)

    def save_crops(
        self,
        crops: Sequence[np.ndarray],
        output_path: str,
        labels: Sequence[str],
        ext: str = "auto",
    ):
        """
        Save the crops to image files.
        :param crops: the crops to save. Will save as RGB if 3 channel [h, w, 3] or
        grayscale if 1 channel [h, w] or [h, w, 1].
        :param labels: the labels to append to the file name, usually the channel names
        associated with each crop.
        :param output_path: the folder to save the crops to. Will make if needed.
        :param ext: the file extension to save the crops as. Defaults to "auto", which
        will save as .tif for grayscale images and .jpg for RGB images.
        :return: None
        """
        if len(crops) != len(labels):
            raise ValueError("Crops and labels must be the same length")

        if csi_images is None or imageio is None:
            raise ModuleNotFoundError(
                "imageio libraries not installed! "
                "run `pip install csi_images[imageio]` to resolve."
            )

        os.makedirs(output_path, exist_ok=True)

        for crop, label in zip(crops, labels):
            if ext == "auto":
                if len(crop.shape) == 2 or crop.shape[2] == 1:
                    file_extension = ".tif"
                elif crop.shape[2] == 3:
                    file_extension = ".jpg"
                else:
                    warnings.warn(
                        f"Image shape {crop.shape} not recognized; saving as .tif"
                    )
                    file_extension = ".tif"
            else:
                file_extension = ext
            file = os.path.join(output_path, f"{self}-{label}{file_extension}")
            # TODO: add more file types here
            if file_extension == ".tif":
                imageio.imwrite(file, crop, compression="deflate")
            elif file_extension in [".jpg", ".jpeg"]:
                crop = csi_images.scale_bit_depth(crop, np.uint8)
                imageio.imwrite(file, crop, quality=80)
            else:
                imageio.imwrite(file, crop)

    def load_crops(
        self, input_path: str, labels: list[str] = None
    ) -> dict[str, np.ndarray]:
        """
        Loads previously saved crop files from a folder.
        :param input_path: folder containing crop files.
        :param labels: optional label filter, will only return crops with these labels.
        :return: a tuple of lists containing the crops and their labels.
        """
        crops = {}
        for file in glob.glob(os.path.join(input_path, f"{self}-*")):
            label = os.path.splitext(os.path.basename(file))[0].split("-")[-1]
            # Skip if we have labels to target
            if labels is not None and label not in labels:
                continue
            crops[label] = imageio.imread(file)
        return crops

    def get_montage_channels(
        self,
        channels: Sequence[int | str] | None,
        composites: dict[int | str, tuple[float, float, float]] | None,
    ) -> tuple[list[int], list[int], dict[int, tuple[float, float, float]]]:
        """
        Get the channel names for the montage from the event's tile.
        :param channels: channel indices or names for grayscale channels
        :param composites: dictionary of channel indices or names and RGB values
        :return: (1) channel indices to retrieve,
                 (2) relative grayscale channel indices, and
                 (3) composite channel indices and RGB values.
        """
        if channels is None:
            channels = list(range(len(self.tile.scan.channels)))
        if (len(channels) == 0) and (composites is None or len(composites) == 0):
            raise ValueError("Must provide at least one channel type to montage")

        channels_to_get = []

        # Build the list of channels to retrieve
        if channels is not None:
            if isinstance(channels[0], str):
                channels = self.tile.scan.get_channel_indices(channels)
            channels_to_get += channels
            order = list(range(len(channels)))  # Always the first n channels
        else:
            order = None

        if composites is not None:
            relative_composites = {}  # Relative indices for retrieved channels
            # Convert to scan indices
            rgb_channels = list(composites.keys())
            if isinstance(rgb_channels[0], str):
                rgb_channels = self.tile.scan.get_channel_indices(rgb_channels)
            # Find the index or add to the end
            for channel, rgb in zip(rgb_channels, composites.values()):
                if channel not in channels_to_get:
                    channels_to_get.append(channel)
                    relative_composites[channel] = rgb
                else:
                    relative_composites[channels_to_get.index(channel)] = rgb
        else:
            relative_composites = None

        return channels_to_get, order, relative_composites

    def get_montage(
        self,
        channels: Sequence[int | str] = None,
        composites: dict[int | str, tuple[float, float, float]] = None,
        mask: np.ndarray[np.uint8] = None,
        labels: Sequence[str] = None,
        crop_size: int = 100,
        in_pixels: bool = True,
        input_path: str = None,
        apply_gain: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Convenience function for getting frame images and creating a montage. Mirrors
        csi_images.make_montage(). Convenient for a single event's montage, but less
        efficient when for multiple events from the same tile.
        :param channels: the channels to use for black-and-white montages.
        :param composites: dictionary of indices and RGB tuples for a composite.
        :param mask: a mask to apply to the montage. Must be the same size as the crop.
        :param crop_size: the square size of the image crop to get for this event.
        :param in_pixels: whether the crop size is in pixels or micrometers. Defaults to pixels.
        :param input_path: the path to the input images. Defaults to None (uses the scan's path).
        :param apply_gain: whether to apply scanner-calculated gain to the images, if
        not already applied. If a list, matches the channels.
        :param kwargs: montage options. See csi_images.make_montage() for more details.
        :return: numpy array representing the montage.
        """
        channels, order, composites = self.get_montage_channels(channels, composites)
        images = self.get_crops(crop_size, in_pixels, input_path, channels, apply_gain)
        return csi_images.make_montage(
            images, order, composites, mask, labels, **kwargs
        )

    def save_montage(
        self,
        montage: np.ndarray,
        output_path: str,
        ocular_names: bool = False,
        tag: str = "",
        file_extension: str = ".jpeg",
        **kwargs,
    ):
        """
        Save the montage as a JPEG image with a set name.
        :param montage: the montage to save.
        :param output_path: the folder to save the montage in. Will make if needed.
        :param ocular_names: whether to use the OCULAR naming convention.
        :param tag: a tag to append to the file name. Ignored if ocular_names is True.
        :param file_extension: the file extension to save the montage as. Defaults to .jpeg.
        :param kwargs: additional arguments to pass to imageio.imwrite().
        :return: None
        """
        if csi_images is None or imageio is None:
            raise ModuleNotFoundError(
                "imageio libraries not installed! "
                "run `pip install csi_images[imageio]` to resolve."
            )

        montage = csi_images.scale_bit_depth(montage, np.uint8)

        if not file_extension.startswith("."):
            file_extension = f".{file_extension}"

        if ocular_names:
            if "cell_id" not in self.metadata.index:
                raise ValueError(
                    "Event metadata must include 'cell_id' for OCULAR naming."
                )
            file = f"{self.tile.n}-{self.metadata['cell_id']}-{self.x}-{self.y}{file_extension}"
        else:
            file = f"{self}{tag}{file_extension}"

        os.makedirs(output_path, exist_ok=True)
        imageio.imwrite(os.path.join(output_path, file), montage, **kwargs)

    def load_montage(self, input_path: str, tag: str = "") -> np.ndarray:
        """
        Loads the montage from a file saved by Event.save_montage.
        :param input_path: the path to the folder where the montage was saved.
        :param tag: a string to add to the file name, before the extension.
        :return:
        """
        file = f"{self}{tag}.jpeg"
        return imageio.imread(os.path.join(input_path, file))

    @classmethod
    def get_many_crops(
        cls,
        events: Sequence[Self],
        crop_size: int | Sequence[int] = 100,
        in_pixels: bool = True,
        input_path: str | Sequence[str] = None,
        channels: Sequence[int | str] = None,
        apply_gain: bool | Sequence[bool] = True,
    ) -> list[list[np.ndarray]]:
        """
        Get the crops for a list of events, ensuring that there is no wasteful reading
        of the same tile multiple times. This function is more efficient than calling
        get_crops() for each event.
        :param events: the events to get crops for.
        :param crop_size: the square size of the image crop to get for this event.
                          Defaults to four times the size of the event.
        :param in_pixels: whether the crop size is in pixels or micrometers.
                          Defaults to pixels, and is ignored if crop_size is None.
        :param input_path: the path to the input images. Will only work for lists of events
                           from the same scan. Defaults to None (uses the scan's path).
        :param channels: the channels to extract images for. Defaults to all channels.
        :param apply_gain: whether to apply scanner-calculated gain to the images, if not already applied. Defaults to True.
                           Can be supplied as a list to apply gain to individual channels.
        :return: a list of lists of cropped images for each event.
        """
        if len(events) == 0:
            return []
        # Adapt singular inputs to lists of appropriate length
        if isinstance(crop_size, int):
            crop_size = [crop_size] * len(events)
        if input_path is None or isinstance(input_path, str):
            input_path = [input_path] * len(events)

        # Get the order of the events when sorted by slide/tile
        order, _ = zip(*sorted(enumerate(events), key=lambda x: x[1].__repr__()))

        # Allocate the list to size
        crops = [[]] * len(events)
        last_tile = None
        images = None  # Holds large numpy arrays, so expensive to compare
        # Iterate through in slide/tile sorted order
        for i in order:
            if last_tile != events[i].tile:
                # Gather the frame images, preserving them for the next event
                frames = Frame.get_frames(events[i].tile, channels)
                if isinstance(apply_gain, bool):
                    apply = [apply_gain] * len(frames)
                else:
                    apply = apply_gain
                images = [f.get_image(input_path[i], a) for f, a in zip(frames, apply)]
                last_tile = events[i].tile
            # Use the frame images to crop the event images
            crops[i] = events[i].crop(images, crop_size[i], in_pixels)
        return crops

    @classmethod
    def get_many_montages(
        cls,
        events: Sequence[Self],
        channels: Sequence[int | str] = None,
        composites: dict[int | str, tuple[float, float, float]] = None,
        masks: Sequence[np.ndarray[np.uint8]] = None,
        labels: Sequence[str] = None,
        crop_size: int = 100,
        in_pixels: bool = True,
        input_path: str = None,
        apply_gain: bool | Iterable[bool] = True,
        **kwargs,
    ) -> list[np.ndarray]:
        """
        Convenience function for get_montage(), but for a list of events. More efficient
        than get_montage() when working with multiple events from the same tile.
        :param events: a list of Event objects.
        :param channels: the channels to extract images for. Defaults to all channels.
        :param composites: dictionary of indices and RGB tuples for a composite.
        :param masks: a list of masks to apply to the montages. Must be the same size as the crops.
        :param labels: the labels to subtitle montage images, usually the channel names
        :param crop_size: the square size of the image crop to get for this event.
        :param in_pixels: whether the crop size is in pixels or micrometers. Defaults to pixels.
        :param input_path: the path to the input images. Defaults to None (uses the scan's path).
        :param apply_gain: whether to apply scanner-calculated gain to the images, if
        not already applied. If a list, matches the channels.
        :param kwargs: montage options. See csi_images.make_montage() for more details.
        :return: a list of numpy arrays representing the montages.
        """
        if len(events) == 0:
            return []
        # Adapt singular inputs to lists of appropriate length
        if isinstance(crop_size, int):
            crop_size = [crop_size] * len(events)
        if input_path is None or isinstance(input_path, str):
            input_path = [input_path] * len(events)
        if masks is None or isinstance(masks, np.ndarray):
            masks = [masks] * len(events)

        # Get the order of the events when sorted by slide/tile
        event_order, _ = zip(*sorted(enumerate(events), key=lambda x: x[1].__repr__()))

        # Allocate the list to size
        montages = [np.empty(0)] * len(events)
        # Placeholder variables to avoid rereading the same tile
        images = None  # Holds large numpy arrays, so expensive to compare
        order = None
        rel_composites = None
        last_tile = None
        # Iterate through in slide/tile sorted order
        for i in event_order:
            if last_tile != events[i].tile:
                channels_to_get, order, rel_composites = events[i].get_montage_channels(
                    channels, composites
                )
                # Gather the frame images, preserving them for the next event
                frames = Frame.get_frames(events[i].tile, channels_to_get)
                if isinstance(apply_gain, bool):
                    apply = [apply_gain] * len(frames)
                else:
                    apply = apply_gain
                images = [f.get_image(input_path[i], a) for f, a in zip(frames, apply)]
                last_tile = events[i].tile
            # Use the frame images to crop the event images and make montages
            crops = events[i].crop(images, crop_size[i], in_pixels)
            montages[i] = csi_images.make_montage(
                crops, order, rel_composites, masks[i], labels, **kwargs
            )

        return montages

    @classmethod
    def get_and_save_many_crops(
        cls,
        events: list[Self],
        output_path: str,
        labels: Sequence[str],
        ext: str = "auto",
        additional_gain: Sequence[float] = None,
        **kwargs,
    ) -> None:
        """
        Get and save the crops for a list of events, ensuring that there is no wasteful
        reading and limiting the image data in memory to 1 tile at a time. This function
        is more efficient that chaining get_crops() and save_crops() for each event or
        get_many_crops() and then save_crops().
        :param events: list of events to get, crop, and save.
        :param output_path: the folder to save the crops in. Will make if needed.
        :param labels: the labels to save the crops with. See save_crops().
        :param ext: the file extension to save the crops as. See save_crops().
        :param additional_gain: additional gain to apply to the crops. If not None, must
        match the length of the number of crop channels.
        :param kwargs: see get_many_crops() for more parameters.
        :return:
        """
        unique_tiles = set([event.tile for event in events])

        for tile in unique_tiles:
            # Get one tile's worth of event crops
            tile_events = [e for e in events if e.tile == tile]
            crops_list = cls.get_many_crops(tile_events, **kwargs)
            for event, crops in zip(tile_events, crops_list):
                # Apply any additional gains
                if additional_gain is not None:
                    crops = [gain * crop for gain, crop in zip(additional_gain, crops)]
                event.save_crops(crops, output_path, labels, ext)

    @classmethod
    def get_and_save_many_montages(
        cls,
        events: list[Self],
        output_path: str,
        ocular_names: bool = False,
        tag: str = "",
        **kwargs,
    ) -> None:
        """
        Save montages of the events to image files.
        :param events: the events to get, montage, and save.
        :param output_path: the folder to save the montages to. Will make if needed.
        :param ocular_names: whether to use the OCULAR naming convention.
        :param tag: a tag to append to the file name. Ignored if ocular_names is True.
        :param kwargs: see get_many_montages() for more parameters.
        """
        unique_tiles = set([event.tile for event in events])

        for tile in unique_tiles:
            # Get one tile's worth of event crops
            tile_events = [e for e in events if e.tile == tile]
            montages = cls.get_many_montages(tile_events, **kwargs)
            for event, montage in zip(tile_events, montages):
                event.save_montage(montage, output_path, ocular_names, tag)


class EventArray:
    """
    A class that holds a large number of events' data, making it easy to analyze and
    manipulate many events at once. A more separated version of the Event class.
    """

    INFO_COLUMNS = ["slide_id", "tile", "roi", "x", "y"]

    def __init__(
        self,
        info: pd.DataFrame = None,
        metadata: pd.DataFrame = None,
        features: pd.DataFrame = None,
    ):
        # Info must be a DataFrame with columns "slide_id", "tile", "roi", "x", "y"
        if info is not None:
            # Special case: "roi" is often not required, so we'll fill in if its missing
            if "roi" not in info.columns:
                info["roi"] = 0
            if set(info.columns) != set(self.INFO_COLUMNS):
                raise ValueError(
                    f"EventArray.info must have columns:"
                    f"{self.INFO_COLUMNS}; had {list(info.columns)}"
                )
            # Copy first to avoid modifying the original
            info = info.copy()
            # Ensure that the columns are the right types
            info["slide_id"] = info["slide_id"].astype(str)
            info["tile"] = info["tile"].astype(np.uint16)
            info["roi"] = info["roi"].astype(np.uint8)
            info["x"] = info["x"].round().astype(np.uint16)
            info["y"] = info["y"].round().astype(np.uint16)
            # Ensure that the columns are in the right order
            info = info[self.INFO_COLUMNS]
        # All DataFrames must all have the same number of rows
        if metadata is not None and (info is None or len(info) != len(metadata)):
            raise ValueError(
                "If EventArray.metadata is not None, it should match rows with .info"
            )
        if features is not None and (info is None or len(info) != len(features)):
            raise ValueError(
                "If EventArray.features is not None, it should match rows with .info"
            )
        # No columns named "metadata_", "features_", or "None"
        column_names = []
        if metadata is not None:
            column_names += metadata.columns.tolist()
        if features is not None:
            column_names += features.columns.tolist()
        if any([col.lower().startswith("metadata_") for col in column_names]):
            raise ValueError("EventArray column names cannot start with 'metadata_'")
        if any([col.lower().startswith("features_") for col in column_names]):
            raise ValueError("EventArray column names cannot start with 'features_'")
        if any([col.lower() == "none" for col in column_names]):
            raise ValueError("EventArray column names cannot be 'none'")

        self.info = info
        self.metadata = metadata
        self.features = features

    def __len__(self) -> int:
        # Convenience method to get the number of events
        if self.info is None:
            return 0
        else:
            return len(self.info)

    def __eq__(self, other):
        # Parse all possibilities for info
        if isinstance(self.info, pd.DataFrame):
            if isinstance(other.info, pd.DataFrame):
                if not self.info.equals(other.info):
                    return False
            else:
                return False
        elif self.info is None:
            if other.info is not None:
                return False

        # Parse all possibilities for metadata
        if isinstance(self.metadata, pd.DataFrame):
            if isinstance(other.metadata, pd.DataFrame):
                is_equal = self.metadata.equals(other.metadata)
                if not is_equal:
                    return False
            else:
                return False
        elif self.metadata is None:
            if other.metadata is not None:
                return False

        # Parse all possibilities for features
        if isinstance(self.features, pd.DataFrame):
            if isinstance(other.features, pd.DataFrame):
                is_equal = self.features.equals(other.features)
                if not is_equal:
                    return False
            else:
                return False
        elif self.features is None:
            if other.features is not None:
                return False

        return is_equal

    def get_sort_order(
        self, by: Hashable | Sequence[Hashable], ascending: bool | Sequence[bool] = True
    ):
        """
        Get the sort order for the EventArray by a column in the info, metadata, or features DataFrames.
        :param by: name of the column(s) to sort by.
        :param ascending: whether to sort in ascending order; can be a list to match by
        :return: the order of the indices to sort by.
        """
        columns = self.get(by)
        return columns.sort_values(by=by, ascending=ascending).index

    def sort(
        self,
        by: Hashable | Sequence[Hashable],
        ascending: bool | Sequence[bool] = True,
    ) -> Self:
        """
        Sort the EventArray by column(s) in the info, metadata, or features DataFrames.
        :param by: name of the column(s) to sort by.
        :param ascending: whether to sort in ascending order; can be a list to match by
        :return: a new, sorted EventArray.
        """
        order = self.get_sort_order(by, ascending)
        info = self.info.loc[order].reset_index(drop=True)
        if self.metadata is not None:
            metadata = self.metadata.loc[order].reset_index(drop=True)
        else:
            metadata = None
        if self.features is not None:
            features = self.features.loc[order].reset_index(drop=True)
        else:
            features = None
        return EventArray(info, metadata, features)

    def get(self, column_names: Hashable | Sequence[Hashable]) -> pd.DataFrame:
        """
        Get a DataFrame with the specified columns from the EventArray, by value.
        :param column_names: the names of the columns to get.
        :return: a DataFrame with the specified columns.
        """
        if isinstance(column_names, Hashable):
            column_names = [column_names]  # Drop into a list for the loop
        columns = []
        for column_name in column_names:
            if column_name in self.info.columns:
                columns.append(self.info[column_name])
            elif self.metadata is not None and column_name in self.metadata.columns:
                columns.append(self.metadata[column_name])
            elif self.features is not None and column_name in self.features.columns:
                columns.append(self.features[column_name])
            else:
                raise ValueError(f"Column {column_name} not found in EventArray")
        return pd.concat(columns, axis=1)

    def rows(self, rows: Sequence[Hashable]) -> Self:
        """
        Get a subset of the EventArray rows based on a boolean or integer index, by value.
        :param rows: row labels, indices, or boolean mask; anything for .loc[]
        :return: a new EventArray with the subset of events.
        """
        info = self.info.loc[rows].reset_index(drop=True)
        if self.metadata is not None:
            metadata = self.metadata.loc[rows].reset_index(drop=True)
        else:
            metadata = None
        if self.features is not None:
            features = self.features.loc[rows].reset_index(drop=True)
        else:
            features = None
        return EventArray(info, metadata, features)

    def copy(self) -> Self:
        """
        Create a deep copy of the EventArray.
        :return: a deep copy of the EventArray.
        """
        return EventArray(
            info=self.info.copy(),
            metadata=None if self.metadata is None else self.metadata.copy(),
            features=None if self.features is None else self.features.copy(),
        )

    # TODO: add a "filter" convenience function that takes a column name and values to filter by

    def add_metadata(self, new_metadata: pd.Series | pd.DataFrame) -> None:
        """
        Add metadata to the EventArray. Removes the need to check if metadata is None.
        Overwrites any existing metadata with the same column names as the new metadata.
        :param new_metadata: the metadata to add.
        """
        if len(self) != len(new_metadata):
            raise ValueError("New metadata must match length of existing info")

        if self.metadata is None:
            self.metadata = new_metadata
        else:
            if isinstance(new_metadata, pd.Series):
                self.metadata[new_metadata.name] = new_metadata
            else:
                # It's a DataFrame
                self.metadata[new_metadata.columns] = new_metadata

    def add_features(self, new_features: pd.Series | pd.DataFrame) -> None:
        """
        Add features to the EventArray. Removes the need to check if features is None.
        Overwrites any existing features with the same column names as the new features.
        :param new_features: the features to add.
        """
        if len(self) != len(new_features):
            raise ValueError("New features must match length of existing info")

        if self.features is None:
            self.features = new_features
        else:
            if isinstance(new_features, pd.Series):
                self.features[new_features.name] = new_features
            else:
                # It's a DataFrame
                self.features[new_features.columns] = new_features

    @classmethod
    def merge(cls, events: Iterable[Self]) -> Self:
        """
        Combine EventArrays in a list into a single EventArray.
        :param events: the new list of events.
        """
        all_info = []
        all_metadata = []
        all_features = []
        for event_array in events:
            # Skip empty EventArrays
            if event_array.info is not None:
                all_info.append(event_array.info)
            if event_array.metadata is not None:
                all_metadata.append(event_array.metadata)
            if event_array.features is not None:
                all_features.append(event_array.features)
        if len(all_info) == 0:
            return EventArray()
        else:
            all_info = pd.concat(all_info, ignore_index=True)
        if len(all_metadata) == 0:
            all_metadata = None
        else:
            all_metadata = pd.concat(all_metadata, ignore_index=True)
        if len(all_features) == 0:
            all_features = None
        else:
            all_features = pd.concat(all_features, ignore_index=True)

        return EventArray(all_info, all_metadata, all_features)

    def to_events(
        self,
        scans: Scan | Iterable[Scan],
        ignore_missing_scans=True,
        ignore_metadata=False,
        ignore_features=False,
    ) -> list[Event]:
        """
        Get the events in the EventArray as a list of events. Returns [] if empty.
        :param scans: the scans that the events belong to, auto-matched by slide_id.
        Pass None if you don't care about scan metadata (pass ignore_missing_scans).
        :param ignore_missing_scans: whether to create blank scans for events without scans.
        :param ignore_metadata: whether to ignore metadata or not
        :param ignore_features: whether to ignore features or not
        :return:
        """
        if len(self) == 0:
            return []
        if isinstance(scans, Scan):
            scans = [scans]
        scans = {scan.slide_id: scan for scan in scans}
        events = []
        for i in range(len(self.info)):
            # Determine the associated scan
            slide_id = self.info["slide_id"][i]
            if slide_id not in scans:
                if ignore_missing_scans:
                    # Create a placeholder scan if the scan is missing
                    scan = Scan.make_placeholder(
                        slide_id,
                        self.info["tile"][i],
                        self.info["roi"][i],
                    )
                else:
                    raise ValueError(
                        f"Scan {self.info['slide_id'][i]} not found for event {i}."
                    )
            else:
                scan = scans[slide_id]

            # Prepare the metadata and features
            if ignore_metadata or self.metadata is None:
                metadata = None
            else:
                # This Series creation method is less efficient,
                # but required for preserving dtypes
                metadata = pd.Series(
                    {col: self.metadata.loc[i, col] for col in self.metadata.columns},
                    dtype=object,
                )
            if ignore_features or self.features is None:
                features = None
            else:
                features = pd.Series(
                    {col: self.features.loc[i, col] for col in self.features.columns},
                    dtype=object,
                )
            # Create the event and append it to the list
            events.append(
                Event(
                    Tile(scan, self.info["tile"][i], self.info["roi"][i]),
                    self.info["x"][i],
                    self.info["y"][i],
                    metadata=metadata,
                    features=features,
                )
            )
        return events

    @classmethod
    def from_events(cls, events: Iterable[Event]) -> Self:
        """
        Set the events in the EventArray to a new list of events.
        :param events: the new list of events.
        """
        info = pd.DataFrame(
            {
                "slide_id": [event.tile.scan.slide_id for event in events],
                "tile": [event.tile.n for event in events],
                "roi": [event.tile.n_roi for event in events],
                "x": [event.x for event in events],
                "y": [event.y for event in events],
            }
        )
        metadata_list = [event.metadata for event in events]
        # Iterate through and ensure that all metadata is the same shape
        for metadata in metadata_list:
            if type(metadata) != type(metadata_list[0]):
                raise ValueError("All metadata must be the same type.")
            if metadata is not None and metadata.shape != metadata_list[0].shape:
                raise ValueError("All metadata must be the same shape.")
        if metadata_list[0] is None:
            metadata = None
        else:
            metadata = pd.DataFrame(metadata_list)
        features_list = [event.features for event in events]
        # Iterate through and ensure that all features are the same shape
        for features in features_list:
            if type(features) != type(features_list[0]):
                raise ValueError("All features must be the same type.")
            if features is not None and features.shape != features_list[0].shape:
                raise ValueError("All features must be the same shape.")
        if features_list[0] is None:
            features = None
        else:
            features = pd.DataFrame(features_list)
        return EventArray(info=info, metadata=metadata, features=features)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert all the data in the EventArray to a single DataFrame.
        :return: a DataFrame with all the data in the EventArray.
        """
        # Make a copy of the info DataFrame and prepend "info_" to the column names
        output = self.info.copy()
        # Combine with the metadata and prepend "metadata_" to the column names
        if self.metadata is not None:
            metadata = self.metadata.copy()
            metadata.columns = [f"metadata_{col}" for col in metadata.columns]
            output = pd.concat([output, metadata], axis=1)
        # Combine with the features and prepend "features_" to the column names
        if self.features is not None:
            features = self.features.copy()
            features.columns = [f"features_{col}" for col in features.columns]
            output = pd.concat([output, features], axis=1)
        return output

    @classmethod
    def from_dataframe(
        cls, df, metadata_prefix: str = "metadata_", features_prefix: str = "features_"
    ) -> Self:
        """
        From a single, special DataFrame, create an EventArray.
        :param df: the DataFrame to convert to an EventArray.
        :param metadata_prefix: the prefix for metadata columns.
        :param features_prefix: the prefix for features columns.
        :return: a DataFrame with all the data in the EventArray.
        """
        # Split the columns into info, metadata, and features and strip prefix
        info = df[[col for col in df.columns if col in cls.INFO_COLUMNS]].copy()
        if info.size == 0:
            info = None
        metadata = df[[col for col in df.columns if col.startswith("metadata_")]].copy()
        metadata.columns = [
            col.replace(metadata_prefix, "") for col in metadata.columns
        ]
        if metadata.size == 0:
            metadata = None
        features = df[[col for col in df.columns if col.startswith("features_")]].copy()
        features.columns = [
            col.replace(features_prefix, "") for col in features.columns
        ]
        if features.size == 0:
            features = None
        return cls(info=info, metadata=metadata, features=features)

    @classmethod
    def from_mask(
        cls,
        mask: np.ndarray,
        tile: Tile,
        include_cell_id: bool = True,
        images: list[np.ndarray] = None,
        image_labels: list[str] = None,
        properties: list[str] = None,
    ) -> Self:
        """
        Extract events from a mask DataFrame, including metadata and features.
        :param mask: the mask to extract events from.
        :param tile: the Tile object associated with this mask.
        :param include_cell_id: whether to include the cell_id, or numerical
        mask label, as metadata in the EventArray.
        :param images: the intensity images to extract features from.
        :param image_labels: the labels for the intensity images.
        :param properties: list of properties to extract in addition to the defaults:
        :return: EventArray corresponding to the mask labels.
        """
        if csi_images is None:
            raise ModuleNotFoundError(
                "imageio libraries not installed! "
                "run `pip install csi_images[imageio]` to resolve."
            )
        # Gather mask_info
        if images is not None and image_labels is not None:
            if len(images) != len(image_labels):
                raise ValueError("Intensity images and labels must match lengths.")

        mask_info = csi_images.extract_mask_info(mask, images, image_labels, properties)

        if len(mask_info) == 0:
            return EventArray()

        # Combine provided info and mask info
        info = pd.DataFrame(
            {
                "slide_id": tile.scan.slide_id,
                "tile": tile.n,
                "roi": tile.n_roi,
                "x": mask_info["x"],
                "y": mask_info["y"],
            },
        )
        # Extract a metadata column if desired
        if include_cell_id:
            metadata = pd.DataFrame({"cell_id": mask_info["id"]})
        else:
            metadata = None
        # If any additional properties were extracted, add them as features
        mask_info = mask_info.drop(columns=["id", "x", "y"], errors="ignore")
        if len(mask_info.columns) > 0:
            features = mask_info
            features.columns = [col.lower() for col in features.columns]
        else:
            features = None
        return EventArray(info, metadata, features)

    def save_csv(self, output_path: str) -> bool:
        """
        Save the events to an CSV file, including metadata and features.
        :param output_path:
        :return:
        """
        if not output_path.endswith(".csv"):
            output_path += ".csv"
        self.to_dataframe().to_csv(output_path, index=False)
        return os.path.exists(output_path)

    @classmethod
    def load_csv(
        cls,
        input_path: str,
        metadata_prefix: str = "metadata_",
        features_prefix: str = "features_",
    ) -> Self:
        """
        Load the events from an CSV file, including metadata and features.
        :param input_path:
        :param metadata_prefix:
        :param features_prefix:
        :return:
        """
        # Load the CSV file
        df = pd.read_csv(input_path)
        return cls.from_dataframe(df, metadata_prefix, features_prefix)

    def save_hdf5(self, output_path: str) -> bool:
        """
        Save the events to an HDF5 file, including metadata and features.
        Uses the pandas-provided HDF5 functions for ease, and external compatibility,
        though these files are slightly harder to view in HDFView or similar.
        :param output_path:
        :return:
        """
        if not output_path.endswith(".hdf5") and not output_path.endswith(".h5"):
            output_path += ".hdf5"
        # Open the output_path as an HDF5 file
        with pd.HDFStore(output_path) as store:
            # Store the dataframes in the HDF5 file
            if self.info is not None:
                store.put("info", self.info, index=False)
            if self.metadata is not None:
                store.put("metadata", self.metadata, index=False)
            if self.features is not None:
                store.put("features", self.features, index=False)
        return os.path.exists(output_path)

    @classmethod
    def load_hdf5(cls, input_path: str) -> Self:
        """
        Load the events from an HDF5 file, including metadata and features.
        :param input_path:
        :return:
        """
        # Open the input_path as an HDF5 file
        with pd.HDFStore(input_path, "r") as store:
            # Load the dataframes from the HDF5 file
            info = store.get("info") if "info" in store else None
            metadata = store.get("metadata") if "metadata" in store else None
            features = store.get("features") if "features" in store else None
        return cls(info=info, metadata=metadata, features=features)

    def save_ocular(self, output_path: str, event_type: str = "cells"):
        """
        Save the events to an OCULAR file. Relies on the dataframe originating
        from an OCULAR file (same columns; duplicate metadata/info).
        :param output_path:
        :param event_type:
        :return:
        """
        if pyreadr is None:
            raise ModuleNotFoundError(
                "pyreadr not installed! Install pyreadr directly "
                "or run `pip install csi-images[rds]` option to resolve."
            )
        if event_type == "cells":
            file_stub = "rc-final"
        elif event_type == "others":
            file_stub = "others-final"
        else:
            raise ValueError("Invalid event type. Must be cells or others.")

        # Ensure good metadata
        metadata = pd.DataFrame(
            {
                "slide_id": self.info["slide_id"],
                "frame_id": self.info["tile"] + 1,  # Convert to 1-indexed for R
                "cell_id": (
                    self.metadata["cell_id"]
                    if "cell_id" in self.metadata.columns
                    else range(len(self.info))
                ),
                "cellx": self.info["x"],
                "celly": self.info["y"],
            }
        )
        if self.metadata is not None:
            metadata[self.metadata.columns] = self.metadata.copy()

        # Check for the "ocular_interesting" column
        if event_type == "cells":
            if "ocular_interesting" in metadata.columns:
                interesting_rows = metadata["ocular_interesting"].to_numpy(dtype=bool)
            elif "hcpc" in metadata.columns:
                # Interesting cells don't get an hcpc designation, leaving them as -1
                interesting_rows = (
                    metadata["hcpc"].to_numpy() == -1
                )  # interesting cells
            else:
                interesting_rows = []
            if sum(interesting_rows) > 0:
                # Split the metadata into interesting and regular
                interesting_events = self.rows(interesting_rows)
                interesting_df = pd.concat(
                    [interesting_events.features, interesting_events.metadata], axis=1
                )
                data_events = self.rows(~interesting_rows)
                data_df = pd.concat(
                    [data_events.features, data_events.metadata], axis=1
                )
                data_df = data_df.drop(columns=["ocular_interesting"], errors="ignore")

                # Drop particular columns for "interesting"
                interesting_df = interesting_df.drop(
                    [
                        "clust",
                        "hcpc",
                        "frame_id",
                        "cell_id",
                        "unique_id",
                        "ocular_interesting",
                    ],
                    axis=1,
                    errors="ignore",
                )
                # Save both .csv and .rds
                interesting_stub = os.path.join(output_path, "ocular_interesting")
                interesting_df.to_csv(f"{interesting_stub}.csv")
                # Suppress pandas FutureWarning
                with warnings.catch_warnings():
                    warnings.simplefilter(action="ignore", category=FutureWarning)
                    pyreadr.write_rds(f"{interesting_stub}.rds", interesting_df)
            else:
                data_df = pd.concat([self.features, metadata], axis=1)
        else:
            # Get all data and reset_index (will copy it)
            data_df = pd.concat([self.features, metadata], axis=1)

        # Split based on cluster number to conform to *-final[1-4].rds
        n_clusters = max(data_df["clust"]) + 1
        split_idx = [round(i * n_clusters / 4) for i in range(5)]
        for i in range(4):
            subset = (split_idx[i] <= data_df["clust"]) & (
                data_df["clust"] < split_idx[i + 1]
            )
            data_df.loc[subset, "hcpc"] = i + 1
            subset = data_df[subset].reset_index(drop=True)
            # Suppress pandas FutureWarning
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=FutureWarning)
                pyreadr.write_rds(
                    os.path.join(output_path, f"{file_stub}{i+1}.rds"), subset
                )

        # Create new example cell strings
        data_df["example_cell_id"] = (
            data_df["slide_id"]
            + " "
            + data_df["frame_id"].astype(str)
            + " "
            + data_df["cell_id"].astype(str)
            + " "
            + data_df["cellx"].astype(int).astype(str)
            + " "
            + data_df["celly"].astype(int).astype(str)
        )
        # Find averagable data columns
        if "cellcluster_id" in data_df.columns:
            end_idx = data_df.columns.get_loc("cellcluster_id")
        else:
            end_idx = data_df.columns.get_loc("slide_id")
        avg_cols = data_df.columns[:end_idx].tolist()
        # Group by cluster and average
        data_df = data_df.groupby("clust").agg(
            **{col: (col, "mean") for col in avg_cols},
            count=("clust", "size"),  # count rows in each cluster
            example_cells=("example_cell_id", lambda x: ",".join(x)),
            hcpc=("hcpc", lambda x: x.iloc[0]),
        )
        data_df = data_df.reset_index()  # Do NOT drop, index is "clust"
        # Create new columns
        metadata = pd.DataFrame(
            {
                "count": data_df["count"],
                "example_cells": data_df["example_cells"],
                "clust": data_df["clust"].astype(int),
                "hcpc": data_df["hcpc"].astype(int),
                "id": data_df["clust"].astype(int).astype(str),
                "cccluster": "0",  # Dummy value
                "ccdistance": 0.0,  # Dummy value
                "rownum": list(range(len(data_df))),
                "framegroup": 0,  # Dummy value
            }
        )
        # Need to pad the features to 761 columns, as per OCULAR report needs
        additional_columns = range(len(avg_cols), 761)
        if len(additional_columns) > 0:
            padding = pd.DataFrame(
                np.zeros((len(data_df), len(additional_columns))),
                columns=[f"pad{i}" for i in additional_columns],
            )
            data_df = pd.concat([data_df[avg_cols], padding, metadata], axis=1)
        else:
            data_df = pd.concat([data_df[avg_cols], metadata], axis=1)

        # Save the cluster data
        data_df.to_csv(os.path.join(output_path, f"{file_stub}.csv"))
        # Suppress pandas FutureWarning
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            pyreadr.write_rds(os.path.join(output_path, f"{file_stub}.rds"), data_df)

    @classmethod
    def load_ocular(
        cls,
        input_path: str,
        event_type="cells",
        cell_data_files=(
            "rc-final1.rds",
            "rc-final2.rds",
            "rc-final3.rds",
            "rc-final4.rds",
            "ocular_interesting.rds",
        ),
        others_data_files=(
            "others-final1.rds",
            "others-final2.rds",
            "others-final3.rds",
            "others-final4.rds",
        ),
        atlas_data_files=(
            "ocular_interesting.rds",
            "ocular_not_interesting.rds",
        ),
        drop_common_events=True,
    ) -> Self:
        """

        :param input_path:
        :param event_type:
        :param cell_data_files:
        :param others_data_files:
        :param atlas_data_files:
        :param drop_common_events:
        :return:
        """
        if pyreadr is None:
            raise ModuleNotFoundError(
                "pyreadr not installed! Install pyreadr directly "
                "or run `pip install csi-images[rds]` option to resolve."
            )
        # Check if the input path is a directory or a file
        if os.path.isfile(input_path):
            data_files = [os.path.basename(input_path)]
            input_path = os.path.dirname(input_path)
        if event_type == "cells":
            data_files = cell_data_files
        elif event_type == "others":
            data_files = others_data_files
        else:
            raise ValueError("Invalid event type.")

        # Load the data from the OCULAR files
        file_data = {}
        for file in data_files:
            file_path = os.path.join(input_path, file)
            if not os.path.isfile(file_path):
                warnings.warn(f"{file} not found for in {input_path}")
                continue
            file_data[file] = pyreadr.read_r(file_path)
            # Get the DataFrame associated with None (pyreadr dict quirk)
            file_data[file] = file_data[file][None]
            if len(file_data[file]) == 0:
                # File gets dropped from the dict
                file_data.pop(file)
                warnings.warn(f"{file} has no cells")
                continue

            # Drop common cells if requested and in this file
            if (
                file in atlas_data_files
                and drop_common_events
                and "catalogue_classification" in file_data[file]
            ):
                common_cell_indices = (
                    file_data[file]["catalogue_classification"] == "common_cell"
                )
                file_data[file] = file_data[file][common_cell_indices == False]

            if len(file_data[file]) == 0:
                # File gets dropped from the dict
                file_data.pop(file)
                warnings.warn(f"{file} has no cells after dropping common cells")
                continue

            # Extract frame_id and cell_id
            # DAPI- events already have frame_id cell_id outside rowname
            if event_type == "cells" and "frame_id" not in file_data[file].columns:
                file_data[file]["rowname"] = file_data[file]["rowname"].astype("str")
                # get frame_id cell_id from rownames column and split into two columns
                split_res = file_data[file]["rowname"].str.split(" ", n=1, expand=True)
                if len(split_res.columns) != 2:
                    warnings.warn(
                        f'Expected "frame_id cell_id" but got {file_data[file]["rowname"]}'
                    )
                # then assign it back to the dataframe
                file_data[file][["frame_id", "cell_id"]] = split_res.astype("int")
            # Ensure frame_id and cell_id are integers
            file_data[file]["frame_id"] = file_data[file]["frame_id"].astype("int")
            file_data[file]["cell_id"] = file_data[file]["cell_id"].astype("int")
            # reset indexes since they can cause NaN values in concat
            file_data[file] = file_data[file].reset_index(drop=True)

        # Merge the data from all files
        if len(file_data) == 0:
            return EventArray()
        elif len(file_data) == 1:
            data = [file_data[file] for file in file_data.keys()][0]
        else:
            data = pd.concat(file_data.values())

        # Others is missing the "slide_id". Insert it right before "frame_id" column
        if event_type == "others" and "slide_id" not in data.columns:
            if os.path.basename(input_path) == "ocular":
                slide_id = os.path.basename(os.path.dirname(input_path))
            else:
                slide_id = "UNKNOWN"
            data.insert(data.columns.get_loc("frame_id"), "slide_id", slide_id)

        # Sort according to ascending cell_id to keep the original, which is in manual_df
        data = data.sort_values(by=["cell_id"], ascending=True)
        # Filter out duplicates by x & y
        data = data.assign(
            unique_id=data["slide_id"]
            + "_"
            + data["frame_id"].astype(str)
            + "_"
            + data["cellx"].astype(int).astype(str)
            + "_"
            + data["celly"].astype(int).astype(str)
        )
        data = data.drop_duplicates(subset=["unique_id"], keep="first")
        # Normal unique_id is with cell_id
        data = data.assign(
            unique_id=data["slide_id"]
            + "_"
            + data["frame_id"].astype(str)
            + "_"
            + data["cell_id"].astype(str)
        )
        data = data.reset_index(drop=True)
        # All columns up to "slide_id" are features; drop the "slide_id"
        features = data.loc[:, :"slide_id"].iloc[:, :-1]
        data = data.loc[:, "slide_id":]
        # Grab the info columns
        info = data[["slide_id", "frame_id", "cellx", "celly"]]
        info.columns = ["slide_id", "tile", "x", "y"]
        info = info.assign(roi=0)  # OCULAR only works on 1 ROI, as far as known
        info = info[["slide_id", "tile", "roi", "x", "y"]]
        # Metadata has duplicate columns for later convenience
        metadata = data
        # Certain columns tend to be problematic with mixed data formats...
        for col in ["TRITC", "CY5", "FITC"]:
            if col in metadata:
                labels = {
                    "False": False,
                    "True": True,
                    "FALSE": False,
                    "TRUE": True,
                    False: False,
                    True: True,
                }
                metadata[col] = metadata[col].map(labels).astype(bool)
        for col in ["catalogue_id", "catalogue_distance", "clust", "hcpc"]:
            if col in metadata:
                metadata[col] = metadata[col].fillna(-1).astype(int)
        info["tile"] = info["tile"] - 1  # Convert to 0-based indexing
        return EventArray(info, metadata, features)
