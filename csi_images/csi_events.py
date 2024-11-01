"""
Contains the Event class, which represents a single event in a scan.
The Event class optionally holds metadata and features. Lists of events with
similar metadata or features can be combined into DataFrames for analysis.

The Event class holds the position of the event in the frame, which can be converted
to the position in the scanner or slide coordinate positions. See the
csi_utils.csi_scans documentation page for more information on the coordinate systems.
"""

import math
import typing

import numpy as np
import pandas as pd
from csi_images import csi_frames, csi_tiles, csi_scans


class Event:
    """
    A class that represents a single event in a scan, making it easy to evaluate
    singular events. Required metadata is exposed as attributes, and optional
    metadata and features are stored as DataFrames.
    """

    # 2D homogenous transformation matrices
    # Translations (final column) are in micrometers (um)
    SCAN_TO_SLIDE_TRANSFORM = {
        csi_scans.Scan.Type.AXIOSCAN7: np.array(
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
        csi_scans.Scan.Type.BZSCANNER: np.array(
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
    imperfections in slides and alignment in the scanners. 
    """

    def __init__(
        self,
        scan: csi_scans.Scan,
        tile: csi_tiles.Tile,
        x: int,
        y: int,
        size: int = 10,  # End-to-end size in pixels
        metadata: pd.Series = None,
        features: pd.Series = None,
    ):
        self.scan = scan
        self.tile = tile
        self.x = x
        self.y = y
        self.size = size
        self.metadata = metadata
        self.features = features

    def __repr__(self) -> str:
        return f"{self.scan.slide_id}-{self.tile.n}-{self.x}-{self.y}"

    def __eq__(self, other) -> bool:
        return self.__repr__() == other.__repr__()

    def get_scan_position(self) -> tuple[float, float]:
        """
        Get the position of the event in the scanner's coordinate frame.
        :return: the scan position of the event in micrometers (um).
        """
        # Get overall pixel position
        pixel_x = self.x + (self.scan.tile_width_px * self.tile.x)
        pixel_y = self.y + (self.scan.tile_height_px * self.tile.y)
        # Convert to micrometers
        x_um = pixel_x * self.scan.pixel_size_um
        y_um = pixel_y * self.scan.pixel_size_um
        # Add the scan's origin in the scanner frame
        x_um += self.scan.roi[self.tile.n_roi].origin_x_um
        y_um += self.scan.roi[self.tile.n_roi].origin_y_um
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
        if self.scan.scanner_id.startswith(self.scan.Type.AXIOSCAN7.value):
            transform = self.SCAN_TO_SLIDE_TRANSFORM[self.scan.Type.AXIOSCAN7]
        elif self.scan.scanner_id.startswith(self.scan.Type.BZSCANNER.value):
            transform = self.SCAN_TO_SLIDE_TRANSFORM[self.scan.Type.BZSCANNER]
        else:
            raise ValueError(f"Scanner type {self.scan.scanner_id} not supported.")
        slide_position = np.matmul(transform, scan_position)
        return float(slide_position[0][0]), float(slide_position[1][0])

    def extract_images(
        self, crop_size: int = 10, in_pixels: bool = False
    ) -> list[np.ndarray]:
        """
        Extract the images from the scan and tile, reading from the file. Called
        "extract" because it must read and extract the images from file, which is slow.
        Use this if you're interested in only a few events, as it is inefficient when
        reading multiple events from the same tile.
        :param crop_size: the square size of the image crop to get for this event. Defaults to 10um.
        :param in_pixels: whether the crop size is in pixels or micrometers. Defaults to um.
        :return: a list of cropped images from the scan in the order of the channels.
        """
        frames = csi_frames.Frame.get_frames(self.tile)
        images = [frame.get_image()[0] for frame in frames]
        return self.crop_images(images, crop_size, in_pixels)

    def crop_images(
        self, images: list[np.ndarray], crop_size: int = 10, in_pixels: bool = False
    ) -> list[np.ndarray]:
        """
        Get the event crops from the frame images. Called "get" because it does not
        need to extract anything; it is very quick for extracting multiple events from
        the same tile.
        Use this if you're interested in many events.
        :param images: the frame images.
        :param crop_size: the square size of the image crop to get for this event. Defaults to 10um.
        :param in_pixels: whether the crop size is in pixels or micrometers. Defaults to um.
        :return: image_size x image_size crops of the event in the provided frames. If
        the event is too close to the edge, the crop will be smaller and not centered.
        """
        # Convert a crop size in micrometers to pixels
        if not in_pixels:
            crop_size = round(crop_size / self.scan.pixel_size_um)
        # Find the crop bounds
        bounds = [
            self.x - crop_size // 2,
            self.y - crop_size // 2,
            self.x + math.ceil(crop_size / 2),
            self.y + math.ceil(crop_size / 2),
        ]
        # Determine how much the bounds violate the image size
        displacements = [
            max(0, -bounds[0]),
            max(0, -bounds[1]),
            max(0, bounds[2] - images[0].shape[1]),
            max(0, bounds[3] - images[0].shape[0]),
        ]
        # Cap off the bounds
        bounds = [
            max(0, bounds[0]),
            max(0, bounds[1]),
            min(images[0].shape[1], bounds[2]),
            min(images[0].shape[0], bounds[3]),
        ]

        for i in range(len(images)):
            # Create a blank image of the right size
            cropped_image = np.zeros((crop_size, crop_size), dtype=np.uint16)

            # Insert the cropped image into the blank image, leaving a black buffer
            # around the edges if the crop would go beyond the original image bounds
            cropped_image[
                displacements[1] : crop_size - displacements[3],
                displacements[0] : crop_size - displacements[2],
            ] = images[i][bounds[1] : bounds[3], bounds[0] : bounds[2]]
            images[i] = cropped_image
        return images


class EventArray:
    """
    A class that holds a large number of events' data, making it easy to analyze and
    manipulate many events at once. A more separated version of the Event class.
    """

    INFO_COLUMNS = ["slide_id", "tile", "roi", "x", "y", "size"]

    def __init__(
        self,
        info: pd.DataFrame = None,
        metadata: pd.DataFrame = None,
        features: pd.DataFrame = None,
    ):
        # Info must be a DataFrame with columns "slide_id", "tile", "roi", "x", "y", "size"
        if info is not None and (
            not all(col in info.columns for col in self.INFO_COLUMNS)
            or len(info.columns) != 6
        ):
            raise ValueError(
                "EventArray.info must have columns 'slide_id', 'tile', 'roi', 'x', 'y', 'size'"
            )
        # All DataFrames must all have the same number of rows
        if metadata is not None and (info is None or len(info) != len(metadata)):
            raise ValueError(
                "If EventArray.metadata is not None, it should match rows with .info"
            )
        if features is not None and (info is None or len(info) != len(features)):
            raise ValueError(
                "If EventArray.features is not None, it should match rows with .info"
            )
        self.info = info
        self.metadata = metadata
        self.features = features

    def __len__(self) -> int:
        # Convenience method to get the number of events
        if self.info is None:
            return 0
        else:
            return len(self.info)

    def add_metadata(self, new_metadata: pd.DataFrame) -> None:
        """
        Add metadata to the EventArray.
        :param new_metadata: the metadata to add.
        """
        if self.metadata is None:
            if len(self) != len(new_metadata):
                raise ValueError("New metadata does not match length of existing info")
            self.metadata = new_metadata
        else:
            # Add the new metadata columns to the existing metadata
            self.metadata = pd.concat([self.metadata, new_metadata], axis=1)

    def add_features(self, new_features: pd.DataFrame) -> None:
        """
        Add features to the EventArray.
        :param new_features: the metadata to add.
        """
        if self.features is None:
            if len(self) != len(new_features):
                raise ValueError("New metadata does not match length of existing info")
            self.features = new_features
        else:
            # Add the new metadata columns to the existing metadata
            self.features = pd.concat([self.features, new_features], axis=1)

    @classmethod
    def from_events(cls, events: list[Event]) -> typing.Self:
        """
        Set the events in the EventArray to a new list of events.
        :param events: the new list of events.
        """
        # Return an empty array if we were passed nothing
        if events is None or len(events) == 0:
            return EventArray()
        # Otherwise, grab the info
        info = pd.DataFrame(
            {
                "slide_id": [event.scan.slide_id for event in events],
                "tile": [event.tile.n for event in events],
                "roi": [event.tile.n_roi for event in events],
                "x": [event.x for event in events],
                "y": [event.y for event in events],
                "size": [event.size for event in events],
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

    def to_events(
        self, scans: list[csi_scans.Scan], ignore_missing_scans=True
    ) -> list[Event]:
        """
        Get the events in the EventArray as a list of events.
        :param scans: the scans that the events belong to.
        :param ignore_missing_scans: whether to create blank scans for events without scans.
        :return:
        """
        events = []
        for i in range(len(self.info)):
            # Determine the associated scan
            scan = None
            for s in scans:
                if s.slide_id == self.info["slide_id"][i]:
                    scan = s
                    break
            if scan is None:
                if ignore_missing_scans:
                    # Create a placeholder scan if the scan is missing
                    scan = csi_scans.Scan.make_placeholder(
                        self.info["slide_id"][i],
                        self.info["tile"][i],
                        self.info["roi"][i],
                    )
                else:
                    raise ValueError(
                        f"Scan {self.info['slide_id'][i]} not found for event {i}."
                    )
            # Add to the list
            events.append(
                Event(
                    scan,
                    csi_tiles.Tile(scan, self.info["tile"][i], self.info["roi"][i]),
                    self.info["x"][i],
                    self.info["y"][i],
                    size=self.info["size"][i],
                    metadata=self.metadata.loc[i],
                    features=self.features.loc[i],
                )
            )
        return events

    @classmethod
    def from_list(cls, events: list[typing.Self]) -> typing.Self:
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


def extract_all_event_images(
    events: list[Event],
    crop_size: list[int] = None,
    in_pixels: bool = False,
) -> list[list[np.ndarray]]:
    """
    Get the images for a list of events, ensuring that there is no wasteful reading
    of the same tile multiple times. This function is more efficient than calling
    extract_event_images for each event.
    TODO: test this function
    :param events: the events to extract images for.
    :param crop_size: the square size of the image crop to get for this event.
                      Defaults to twice the size of the event.
    :param in_pixels: whether the crop size is in pixels or micrometers.
                      Defaults to um.
    :return: a list of lists of cropped images for each event.
    """

    # Sort the events by tile; use a shallow copy to avoid modifying the original
    events, order = zip(*sorted(enumerate(events), key=lambda x: x[1].tile.__repr__()))

    if crop_size is None:
        crop_size = [2 * event.size for event in events]
        in_pixels = True

    # Allocate the list to size
    images = [None] * len(events)
    last_tile = None
    frame_images = None  # Holds large numpy arrays, so expensive to compare
    for i in range(len(events)):
        if last_tile != events[i].tile:
            # Gather the frame images, preserving them for the next event
            frames = csi_frames.get_frames(events[i].tile)
            frame_images = [frame.get_image()[0] for frame in frames]

            last_tile = events[i].tile
        # Use the frame images to crop the event images
        # Preserve the original order using order[i]
        images[order[i]] = events[i].crop_images_to_event(
            frame_images, crop_size[i], in_pixels
        )
    return images


def get_features_as_dataframe(events: list[Event]) -> pd.DataFrame:
    """
    Combine the features of a list of events into a single DataFrame.
    TODO: test this function
    :param events: the events to gather features for.
    :return: a DataFrame with all of the features.
    """

    features = [event.features for event in events]
    if any(feature is None for feature in features):
        raise ValueError("Some events are missing features.")
    elif any(feature.shape != features[0].shape for feature in features):
        raise ValueError("Features are not the same shape.")

    features = pd.concat(features)
    features["descriptor"] = [event.__repr__() for event in events]
    features.set_index("descriptor", inplace=True)
    return features


def get_metadata_as_dataframe(events: list[Event]) -> pd.DataFrame:
    """
    Combine the metadata of a list of events into a single DataFrame.
    TODO: test this function
    :param events: the events to gather features for.
    :return: a DataFrame with all of the event metadata.
    """

    metadatas = [event.metadata for event in events]
    if any(metadata is None for metadata in metadatas):
        raise ValueError("Some events are missing metadata.")
    elif any(metadata.shape != metadata[0].shape for metadata in metadatas):
        raise ValueError("Metadata are not the same shape.")

    metadatas = pd.concat(metadatas)
    metadatas["descriptor"] = [event.__repr__() for event in events]
    metadatas.set_index("descriptor", inplace=True)
    return metadatas


def save_to_hdf5(events: list[Event], output_path: str) -> bool:
    """
    Save the events to an HDF5 file, including metadata and features.
    :param events:
    :param output_path:
    :return:
    """
    raise NotImplementedError("This function is not yet implemented.")


def load_from_hdf5(input_path: str) -> list[Event]:
    """
    Load the events from an HDF5 file, including metadata and features.
    :param input_path:
    :return:
    """
    raise NotImplementedError("This function is not yet implemented.")


def save_to_csv(events: list[Event], output_path: str) -> bool:
    """
    Save the events to an CSV file, including metadata and features.
    :param events:
    :param output_path:
    :return:
    """
    raise NotImplementedError("This function is not yet implemented.")


def load_from_csv(input_path: str) -> list[Event]:
    """
    Load the events from an CSV file, including metadata and features.
    :param input_path:
    :return:
    """
    raise NotImplementedError("This function is not yet implemented.")
