"""
Contains the Event class, which represents a single event in a scan.
The Event class optionally holds metadata and features. Lists of events with
similar metadata or features can be combined into DataFrames for analysis.

The Event class holds the position of the event in the frame, which can be converted
to the position in the scanner or slide coordinate positions. See the
csi_utils.csi_scans documentation page for more information on the coordinate systems.
"""

import os
import math
import typing

import numpy as np
import pandas as pd

import pyreadr

from .csi_scans import Scan
from .csi_tiles import Tile
from .csi_frames import Frame


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
        scan: Scan,
        tile: Tile,
        x: int,
        y: int,
        size: int = 12,  # End-to-end size in pixels
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

    def __lt__(self, other):
        return self.__repr__() < other.__repr__()

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

    def crop_images(
        self, images: list[np.ndarray], crop_size: int = 100, in_pixels: bool = True
    ) -> list[np.ndarray]:
        """
        Get the event crops from the frame images. Called "get" because it does not
        need to extract anything; it is very quick for extracting multiple events from
        the same tile.
        Use this if you're interested in many events.
        :param images: the frame images.
        :param crop_size: the square size of the image crop to get for this event.
        :param in_pixels: whether the crop size is in pixels or micrometers. Defaults to pixels.
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

        # Crop the images
        cropped_images = []
        for image in images:
            # Create a blank image of the right size
            cropped_image = np.zeros((crop_size, crop_size), dtype=image.dtype)

            # Insert the cropped image into the blank image, leaving a black buffer
            # around the edges if the crop would go beyond the original image bounds
            cropped_image[
                displacements[1] : crop_size - displacements[3],
                displacements[0] : crop_size - displacements[2],
            ] = image[bounds[1] : bounds[3], bounds[0] : bounds[2]]
            cropped_images.append(cropped_image)
        return cropped_images

    def extract_images(
        self, crop_size: int = 100, in_pixels: bool = True
    ) -> list[np.ndarray]:
        """
        Extract the images from the scan and tile, reading from the file. Called
        "extract" because it must read and extract the images from file, which is slow.
        Use this if you're interested in only a few events, as it is inefficient when
        reading multiple events from the same tile.
        :param crop_size: the square size of the image crop to get for this event.
        :param in_pixels: whether the crop size is in pixels or micrometers. Defaults to pixels.
        :return: a list of cropped images from the scan in the order of the channels.
        """
        frames = Frame.get_frames(self.tile)
        images = [frame.get_image() for frame in frames]
        return self.crop_images(images, crop_size, in_pixels)

    @classmethod
    def extract_images_for_list(
        cls,
        events: list[typing.Self],
        crop_size: int | list[int] = None,
        in_pixels: bool = True,
    ) -> list[list[np.ndarray]]:
        """
        Get the images for a list of events, ensuring that there is no wasteful reading
        of the same tile multiple times. This function is more efficient than calling
        extract_event_images for each event.
        TODO: test this function
        :param events: the events to extract images for.
        :param crop_size: the square size of the image crop to get for this event.
                          Defaults to four times the size of the event.
        :param in_pixels: whether the crop size is in pixels or micrometers.
                          Defaults to pixels, and is ignored if crop_size is None.
        :return: a list of lists of cropped images for each event.
        """
        if len(events) == 0:
            return []

        # Populate a crop size if none provided
        if crop_size is None:
            crop_size = [4 * event.size for event in events]
            in_pixels = True
        # Propagate a constant crop size
        elif isinstance(crop_size, int):
            crop_size = [crop_size] * len(events)

        # Sort the events by tile; use a shallow copy to avoid modifying the original
        order, _ = zip(*sorted(enumerate(events), key=lambda x: x[1].__repr__()))

        # Allocate the list to size
        images = [None] * len(events)
        last_tile = None
        frame_images = None  # Holds large numpy arrays, so expensive to compare
        # Iterate through in sorted order
        for i in order:
            if last_tile != events[i].tile:
                # Gather the frame images, preserving them for the next event
                frames = Frame.get_frames(events[i].tile)
                frame_images = [frame.get_image() for frame in frames]

                last_tile = events[i].tile
            # Use the frame images to crop the event images
            # Preserve the original order using order[i]
            images[i] = events[i].crop_images(frame_images, crop_size[i], in_pixels)
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
            not all(
                col in info.columns
                for col in ["slide_id", "tile", "roi", "x", "y", "size"]
            )
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

    def __eq__(self, other):
        is_equal = True
        # Parse all possibilities for info
        if isinstance(self.info, pd.DataFrame):
            if isinstance(other.info, pd.DataFrame):
                is_equal = self.info.equals(other.info)
                if not is_equal:
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

    def get_sort_order(self, by: str | list[str], ascending: bool | list[bool] = True):
        """
        Get the sort order for the EventArray by a column in the info, metadata, or features DataFrames.
        :param by: name of the column(s) to sort by.
        :param ascending: whether to sort in ascending order; can be a list to match by
        :return: the order of the indices to sort by.
        """
        columns = self.get(by)
        return columns.sort_values(by=by, ascending=ascending).index

    def sort(
        self, by: str | list[str], ascending: bool | list[bool] = True
    ) -> typing.Self:
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

    def get(self, column_names: int | str | list[int] | list[str]) -> pd.DataFrame:
        """
        Get a DataFrame with the specified columns from the EventArray, by value.
        :param column_names: the names of the columns to get.
        :return: a DataFrame with the specified columns.
        """
        if isinstance(column_names, str):
            column_names = [column_names]
        columns = []
        for column_name in column_names:
            if column_name in self.info.columns:
                columns.append(self.info[column_name])
            elif self.metadata is not None and column_name in self.metadata.columns:
                columns.append(column_name)
            elif self.features is not None and column_name in self.features.columns:
                columns.append(column_name)
            else:
                raise ValueError(f"Column {column_name} not found in EventArray")
        return pd.concat(columns, axis=1)

    def rows(self, rows) -> typing.Self:
        """
        Get a subset of the EventArray rows based on a boolean or integer index, by value.
        :param rows: the indices to get as a 1D boolean/integer list/array/series
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

    def copy(self) -> typing.Self:
        """
        Create a deep copy of the EventArray.
        :return: a deep copy of the EventArray.
        """
        return EventArray(
            info=self.info.copy(),
            metadata=None if self.metadata is None else self.metadata.copy(),
            features=None if self.features is None else self.features.copy(),
        )

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
            self.metadata = pd.concat([self.metadata, new_metadata], axis=1, copy=False)

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
    def merge(cls, events: list[typing.Self]) -> typing.Self:
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
        self,
        scans: list[Scan],
        ignore_missing_scans=True,
        ignore_metadata=False,
        ignore_features=False,
    ) -> list[Event]:
        """
        Get the events in the EventArray as a list of events.
        :param scans: the scans that the events belong to. Pass an empty list if you
                      don't care about scan metadata.
        :param ignore_missing_scans: whether to create blank scans for events without scans.
        :param ignore_metadata: whether to ignore metadata or not
        :param ignore_features: whether to ignore features or not
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
                    scan = Scan.make_placeholder(
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
                    Tile(scan, self.info["tile"][i], self.info["roi"][i]),
                    self.info["x"][i],
                    self.info["y"][i],
                    size=self.info["size"][i],
                    metadata=None if ignore_metadata else self.metadata.loc[i],
                    features=None if ignore_features else self.features.loc[i],
                )
            )
        return events

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert all the data in the EventArray to a single DataFrame.
        :return: a DataFrame with all the data in the EventArray.
        """
        # Make a copy of the info DataFrame and prepend "info_" to the column names
        output = self.info.copy()
        output.columns = [f"info_{col}" for col in output.columns]
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
    def from_dataframe(cls, df) -> typing.Self:
        """
        From a single, special DataFrame, create an EventArray.
        :return: a DataFrame with all the data in the EventArray.
        """
        # Split the columns into info, metadata, and features and strip prefix
        info = df[[col for col in df.columns if col.startswith("info_")]].copy()
        info.columns = [col.replace("info_", "") for col in info.columns]
        if info.size == 0:
            info = None
        metadata = df[[col for col in df.columns if col.startswith("metadata_")]].copy()
        metadata.columns = [col.replace("metadata_", "") for col in metadata.columns]
        if metadata.size == 0:
            metadata = None
        features = df[[col for col in df.columns if col.startswith("features_")]].copy()
        features.columns = [col.replace("features_", "") for col in features.columns]
        if features.size == 0:
            features = None
        return cls(info=info, metadata=metadata, features=features)

    def save_csv(self, output_path: str) -> bool:
        """
        Save the events to an CSV file, including metadata and features.
        :param output_path:
        :return:
        """
        self.to_dataframe().to_csv(output_path, index=False)
        return os.path.exists(output_path)

    @classmethod
    def load_csv(cls, input_path: str) -> typing.Self:
        """
        Load the events from an CSV file, including metadata and features.
        :param input_path:
        :return:
        """
        # Load the CSV file
        df = pd.read_csv(input_path)
        return cls.from_dataframe(df)

    def save_hdf5(self, output_path: str) -> bool:
        """
        Save the events to an HDF5 file, including metadata and features.
        Uses the pandas-provided HDF5 functions for ease, and external compatibility,
        though these files are slightly harder to view in HDFView or similar.
        :param output_path:
        :return:
        """
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
    def load_hdf5(cls, input_path: str) -> typing.Self:
        """
        Load the events from an HDF5 file, including metadata and features.
        :param input_path:
        :return:
        """
        # Open the input_path as an HDF5 file
        with pd.HDFStore(input_path) as store:
            # Load the dataframes from the HDF5 file
            info = store.get("info") if "info" in store else None
            metadata = store.get("metadata") if "metadata" in store else None
            features = store.get("features") if "features" in store else None
        return cls(info=info, metadata=metadata, features=features)

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
        merge_event_data_with_stats=True,
        filter_and_generate_morphs=True,
        drop_common_events=True,
        log=None,
    ) -> typing.Self:
        """

        :param input_path:
        :param event_type:
        :param cell_data_files:
        :param others_data_files:
        :param atlas_data_files:
        :param merge_event_data_with_stats:
        :param filter_and_generate_morphs:
        :param drop_common_events:
        :param log:
        :return:
        """
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
                if log is not None:
                    log.warning(f"{file} not found for in {input_path}")
                continue
            file_data[file] = pyreadr.read_r(file_path)
            # Get the DataFrame associated with None (pyreadr dict quirk)
            file_data[file] = file_data[file][None]
            if len(file_data[file]) == 0:
                # File gets dropped from the dict
                file_data.pop(file)
                if log is not None:
                    log.warning(f"{file} has no cells")
                continue

            if log is not None:
                log.debug(f"{file} has {len(file_data[file])} cells")

            # Drop common cells if requested and in this file
            if file in atlas_data_files and drop_common_events:
                common_cell_indices = (
                    file_data[file]["catalogue_classification"] == "common_cell"
                )
                if log is not None:
                    log.debug(
                        f"Dropping {int(pd.Series.sum(common_cell_indices))}"
                        f"common cells from {file}"
                    )
                file_data[file] = file_data[file][common_cell_indices == False]

            if len(file_data[file]) == 0:
                # File gets dropped from the dict
                file_data.pop(file)
                if log is not None:
                    log.warning(f"{file} has no cells after dropping common cells")
                continue

            # Extract frame_id and cell_id
            # DAPI- events already have frame_id cell_id outside rowname
            if event_type == "cells":
                file_data[file]["rowname"] = file_data[file]["rowname"].astype("str")
                # get frame_id cell_id from rownames column and split into two columns
                split_res = file_data[file]["rowname"].str.split(" ", n=1, expand=True)
                if len(split_res.columns) != 2:
                    log.warning(
                        f'Expected "frame_id cell_id" but got {file_data[file]["rowname"]}'
                    )
                # then assign it back to the dataframe
                file_data[file][["frame_id", "cell_id"]] = split_res.astype("int")
            # reset indexes since they can cause NaN values in concat
            file_data[file] = file_data[file].reset_index(drop=True)

        # Merge the data from all files
        if len(file_data) == 0:
            return EventArray()
        elif len(file_data) == 1:
            data = [file_data[file] for file in file_data.keys()][0]
        else:
            data = pd.concat(file_data.values())

        if log is not None:
            log.debug(f"Gathered a total of {len(data)} events")

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
        # Filter out duplicates by cell_id
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
        info = info.assign(
            roi=0,  # OCULAR only works on 1 ROI, as far as known
            size=25,  # Static, for later montaging
        )
        info = info[["slide_id", "tile", "roi", "x", "y", "size"]]
        # Metadata has duplicate columns for later convenience
        metadata = data
        return EventArray(info, metadata, features)

    def save_ocular(self, output_path: str, event_type: str = "cells") -> bool:
        """
        Save the events to an OCULAR file. Relies on the dataframe originating
        from an OCULAR file (same columns; duplicate metadata/info).
        :param output_path:
        :return:
        """
        if event_type == "cells":
            file_stub = "rc-final"
        elif event_type == "others":
            file_stub = "others-final"
        else:
            raise ValueError("Invalid event type. Must be cells or others.")

        # Check for the "ocular_interesting" column
        if event_type == "cells" and "ocular_interesting" in self.metadata.columns:
            interesting = self.metadata["ocular_interesting"]
            # Split the metadata into interesting and regular
            # Interesting will only have dropped columns, with no internal changes
            interesting = pd.concat(
                [self.features[interesting], self.metadata[interesting]], axis=1
            ).reset_index(drop=True)
            # Data will get some columns changed; reset_index will copy it
            data = (
                pd.concat(
                    [self.features[~interesting], self.metadata[~interesting]], axis=1
                )
                .reset_index(drop=True)
                .drop(columns=["ocular_interesting"])
            )

            # Drop particular columns for "interesting"
            interesting = interesting.drop(
                [
                    "clust",
                    "hcpc",
                    "frame_id",
                    "cell_id",
                    "unique_id",
                    "ocular_interesting",
                ],
                axis=1,
            )
            # Save both .csv and .rds
            interesting.to_csv(
                os.path.join(output_path, "ocular_interesting.csv"), index=False
            )
            pyreadr.write_rds(
                os.path.join(output_path, "ocular_interesting.rds"), interesting
            )
        else:
            # Get all data and reset_index (will copy it)
            data = pd.concat([self.features, self.metadata], axis=1).reset_index(
                drop=True
            )

        # Split based on cluster number to conform to *-final[1-4].rds
        n_clusters = max(data["clust"]) + 1
        split_idx = [round(i * n_clusters / 4) for i in range(5)]
        for i in range(4):
            subset = (split_idx[i] <= data["clust"]) & (
                data["clust"] < split_idx[i + 1]
            )
            subset = data[subset].reset_index(drop=True)
            subset["hcpc"] = i + 1
            pyreadr.write_rds(
                os.path.join(output_path, f"{file_stub}{i+1}.rds"), subset
            )

        # Create new example cell strings
        data["example_cell_id"] = (
            data["slide_id"]
            + " "
            + data["frame_id"].astype(str)
            + " "
            + data["cell_id"].astype(str)
            + " "
            + data["cellx"].astype(int).astype(str)
            + " "
            + data["celly"].astype(int).astype(str)
        )
        # Find averagable data columns
        if "cellcluster_id" in data.columns:
            avg_cols = data.columns[: data.columns.get_loc("cellcluster_id")].tolist()
        else:
            avg_cols = data.columns[: data.columns.get_loc("slide_id")].tolist()
        # Group by cluster and average
        data = data.groupby("clust").agg(
            **{col: (col, "mean") for col in avg_cols},
            count=("clust", "size"),  # count rows in each cluster
            example_cells=("example_cell_id", lambda x: ",".join(x)),
            hcpc=("hcpc", lambda x: x.iloc[0]),
        )
        data = data.reset_index()  # Do NOT drop, index is "clust"
        # Create new columns
        metadata = pd.DataFrame(
            {
                "count": data["count"],
                "example_cells": data["example_cells"],
                "clust": data["clust"].astype(int),
                "hcpc": data["hcpc"].astype(int),
                "id": data["clust"].astype(int).astype(str),
                "cccluster": "0",  # Dummy value
                "ccdistance": 0.0,  # Dummy value
                "rownum": list(range(len(data))),
                "framegroup": 0,  # Dummy value
            }
        )
        data = pd.concat([data[avg_cols], metadata], axis=1)
        # Save the data
        data.to_csv(os.path.join(output_path, f"{file_stub}.csv"), index=False)
        pyreadr.write_rds(os.path.join(output_path, f"{file_stub}.rds"), data)
