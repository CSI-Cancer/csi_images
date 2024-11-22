import os

import numpy as np
import pandas as pd

from csi_images.csi_events import EventArray

FRAME_INFO_FILE = "frameinfo.csv"
FRAME_MORPHOMETRICS_FILES = ["framestat-means.csv", "framestat-dev.csv"]
SLIDE_MORPHOMETRICS_FILE = "slidestat-calc.csv"


def get_cells(report_path: str) -> EventArray:
    """
    Convenience function to read the cells (post-clustering) from OCULAR.
    :param report_path:
    :return:
    """
    return EventArray.load_ocular(report_path, event_type="cells")


def get_others(report_path: str) -> EventArray:
    """
    Convenience function to read the DAPI- events (post-clustering from OCULAR.
    :param report_path:
    :return:
    """
    return EventArray.load_ocular(report_path, event_type="others")


def save_cells(report_path: str, events: EventArray):
    """
    Convenience function to save the cells (post-clustering) from OCULAR.
    :param report_path:
    :param events:
    :return:
    """
    return events.save_ocular(report_path, event_type="cells")


def save_others(report_path: str, events: EventArray):
    """
    Convenience function to save the DAPI- events (post-clustering) from OCULAR.
    :param report_path:
    :param events:
    :return:
    """
    return events.save_ocular(report_path, event_type="others")


def get_frame_info(report_path: str) -> pd.DataFrame:
    """
    Reads frameinfo.csv with high-level frame metadata.
    :param report_path: path to the OCULAR report folder.
    :return: DataFrame with frame info
    """
    file_path = os.path.join(report_path, FRAME_INFO_FILE)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"{file_path} not found")
    # Read, dropping the repetitive first column if it exists
    data = pd.read_csv(file_path).drop(columns=["Unnamed: 0"], errors="ignore")
    return data


def get_frame_statistics(report_path: str) -> pd.DataFrame:
    """
    Reads framestat-means.csv and framestat-dev.csv and merges them.
    :param report_path: path to the OCULAR report folder.
    :return:
    """
    # Check for existence of all files
    file_paths = []
    for file in FRAME_MORPHOMETRICS_FILES:
        file_path = os.path.join(report_path, file)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"{file_path} not found")
        file_paths.append(file_path)

    data = []
    # Read in the data from each of the files
    for file in file_paths:
        file_data = pd.read_csv(file)
        # Rename unnamed column to frame_id
        file_data = file_data.rename(columns={"Unnamed: 0": "frame_id"})
        # Add an appropriate prefix to the column names
        if "means" in file:
            prefix_name = "frame_mean_"
        elif "dev" in file:
            prefix_name = "frame_sdev_"
        else:
            # Unexpected file name; no prefix
            prefix_name = ""
        file_data = file_data.add_prefix(prefix_name)
        # Strip the prefix from the frame_id column
        file_data = file_data.rename(columns={prefix_name + "frame_id": "frame_id"})
        data.append(file_data)
    data = pd.merge(data[0], data[1], on="frame_id")
    return data


def get_slide_statistics(report_path: str) -> pd.DataFrame:
    """
    Gets slide-level morphometric statistics from slidestat-calc.csv.
    :param report_path: path to the OCULAR report folder.
    :return:
    """
    file_path = os.path.join(report_path, SLIDE_MORPHOMETRICS_FILE)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"{file_path} not found")

    data = pd.read_csv(file_path)

    # Row 0 is mean; convert to dataframe and transpose
    mean = data.iloc[0, 1:].to_frame().transpose().reset_index(drop=True)
    mean = mean.add_prefix("slide_mean_")

    # Row 1 is standard deviations; convert to dataframe and transpose
    sdev = data.iloc[1, 1:].to_frame().transpose().reset_index(drop=True)
    sdev = sdev.add_prefix("slide_sdev_")

    data = pd.concat([mean, sdev], axis=1)
    return data


def merge_statistics(
    events: EventArray,
    frame_stats: pd.DataFrame,
    slide_stats: pd.DataFrame,
) -> EventArray:
    """
    Merges frame-level and slide-level morphometric statistics into the EventArray.
    :param events: EventArray object
    :param frame_stats: frame-level morphometric statistics
    :param slide_stats: slide-level morphometric statistics
    :return: a new EventArray object with the merged data
    """
    # Create a combined slide and frame stats dataframe (1 + 2*761 columns)
    slide_stats = pd.concat([slide_stats] * len(frame_stats), axis=0, ignore_index=True)
    all_stats = pd.concat([frame_stats, slide_stats], axis=1)

    # Do not modify the original events
    events = events.copy()
    # Check that all event frame_ids are in the stats
    if not set(events.metadata["frame_id"]).issubset(set(all_stats["frame_id"])):
        raise ValueError("Not all frame_ids are present in the morphometric statistics")
    # Merge together using the frame_id, then drop the frame_id column from features
    events.add_features(pd.DataFrame({"frame_id": events.metadata["frame_id"]}))
    # Must be a left join to keep features in the same order! Finding that took me 2h
    events.features = pd.merge(events.features, all_stats, on=["frame_id"], how="left")
    events.features = events.features.drop(columns=["frame_id"])
    return events


def filter_and_generate_statistics(
    events: EventArray,
    morphs_to_drop: list[str] = None,
    morphs_to_keep: list[str] = None,
) -> EventArray:
    """

    :param events:
    :param morphs_to_drop:
    :param morphs_to_keep:
    :return:
    """
    # Do not modify the original events
    events = events.copy()
    # columns to keep are the columns kept after pull
    # initialize that variable to all columns for starters

    channels = {
        "D": "dapi",
        "CK": "tritc",
        "V": "fitc",
        "CD": "cy5",
    }

    if morphs_to_drop is None:
        morphs_to_drop = []
    if morphs_to_keep is None:
        morphs_to_keep = []
    # 'multi_channel' problematic

    # Identify columns that should be kept, considering the morphs_to_drop
    columns_to_keep = list(events.features.columns)
    if "haralick" in morphs_to_drop:
        columns_to_keep = [col for col in columns_to_keep if ".h." not in col]
    # remove theta
    if "theta" in morphs_to_drop:
        columns_to_keep = [col for col in columns_to_keep if "theta" not in col]
    # remove blurred and then extracted
    if "blurred" in morphs_to_drop:
        for channel in channels.keys():
            columns_to_keep = [
                col for col in columns_to_keep if f"B{channels[channel]}" not in col
            ]
    # Remove everything that is not only for one channel
    if "multi_channel" in morphs_to_drop:
        for channel in channels.keys():
            columns_to_keep = [
                col for col in columns_to_keep if f".{channels[channel]}." in col
            ]
    # keep only mean, sd, and median
    if "mean_sd_q05" in morphs_to_keep:
        columns_to_keep = [
            col
            for col in columns_to_keep
            if (".mean" in col) or (".sd" in col) or (".q05" in col)
        ]
    # keep only mean and median
    if "mean_q05" in morphs_to_keep:
        columns_to_keep = [
            col for col in columns_to_keep if (".mean" in col) or (".q05" in col)
        ]
    # remove slide level info
    if "slide" in morphs_to_drop:
        columns_to_keep = [col for col in columns_to_keep if "slide" not in col]
    # remove frame level info
    if "frame" in morphs_to_drop:
        columns_to_keep = [col for col in columns_to_keep if "frame" not in col]
    # drop duplicates
    columns_to_keep = list(set(columns_to_keep))

    cell_features_for_sdom_frame_level = [
        "cellf.tritc.b.mean",
        "cellf.tritc.b.sd",
        "cellf.tritc.b.mad",
        "cellf.tritc.b.q001",
        "cellf.tritc.b.q005",
        "cellf.tritc.b.q05",
        "cellf.tritc.b.q095",
        "cellf.tritc.b.q099",
        "tritc_cy5_ratio",
    ]
    cell_features_for_sdom_frame_level += [
        "cellf.fitc.b.mean",
        "cellf.fitc.b.sd",
        "cellf.fitc.b.mad",
        "cellf.fitc.b.q001",
        "cellf.fitc.b.q005",
        "cellf.fitc.b.q05",
        "cellf.fitc.b.q095",
        "cellf.fitc.b.q099",
    ]
    cell_features_for_sdom_frame_level += [
        "cellf.cy5.b.mean",
        "cellf.cy5.b.sd",
        "cellf.cy5.b.mad",
        "cellf.cy5.b.q001",
        "cellf.cy5.b.q005",
        "cellf.cy5.b.q05",
        "cellf.cy5.b.q095",
        "cellf.cy5.b.q099",
    ]
    cell_features_for_sdom_frame_level += [
        "nucleusf.dapi.b.mean",
        "nucleusf.dapi.b.sd",
        "nucleusf.dapi.b.mad",
        "nucleusf.dapi.b.q001",
        "nucleusf.dapi.b.q005",
        "nucleusf.dapi.b.q05",
        "nucleusf.dapi.b.q095",
        "nucleusf.dapi.b.q099",
    ]
    # Calculate SDOMs
    computed_features = []
    sdom_prefix = "cell_sdom_frame_level_"
    computed_features += [
        sdom_prefix + item for item in cell_features_for_sdom_frame_level
    ]
    events.features = _generate_computed_features(events.features, computed_features)

    columns_to_keep += computed_features
    events.features = events.features[columns_to_keep]

    return events


# computed features is a list containing the features that you want
# e.g. 'cell_sdom_framelevel_'+ cell feature will automatically generate the correct features
def _generate_computed_features(df: pd.DataFrame, features_to_compute: list[str]):
    """
    Calculates SDOMs for the given features, adding them as a column in the dataframe.
    :param df: DataFrame with the features.
    :param features_to_compute: list of features to compute SDOMs for.
    :return:
    """
    # TODO: figure out if there were supposed to be more than 1 type...
    # Type 1: cell_sdom_frame_level_ features:
    feature_prefix = "cell_sdom_frame_level_"
    # identify these computed features
    sdom_frame_features = [s for s in features_to_compute if feature_prefix in s]

    for feature in sdom_frame_features:
        df[feature] = _calculate_sdom(df, feature, sdom_string_prefix=feature_prefix)

    return df


def _calculate_sdom(
    df: pd.DataFrame,
    sdom_frame_feature: str,
    sdom_string_prefix: str = "cell_sdom_frame_level_",
    fill_na=0,
) -> pd.DataFrame:
    # split string
    split_string = sdom_frame_feature.split(sdom_string_prefix)
    # get cell feature
    cell_feature = split_string[1]
    # get corresponding frame feature
    frame_feature = "frame_mean_" + cell_feature
    # get corresponding frame sdev feature
    frame_feature_sd = "frame_sdev_" + cell_feature

    df[sdom_frame_feature] = df[cell_feature] - df[frame_feature]
    # if a frame has only 1 cell then this will divide by zero
    # that is why i use the divide function of pandas
    df[sdom_frame_feature] = df[sdom_frame_feature].divide(df[frame_feature_sd])

    # replace infinity by zero
    df[sdom_frame_feature].replace(np.inf, 0)

    if fill_na is not None and df[sdom_frame_feature].isna().any():
        n_of_nas = df[sdom_frame_feature][df[sdom_frame_feature].isna()].index
        print(
            f"Replacing nan with zero for {sdom_frame_feature} , number of entries: {len(n_of_nas)}"
        )
        df[sdom_frame_feature].fillna(fill_na, inplace=True)

    # TODO: clean this up, it either needs to replace or return (probably return)
    return df[sdom_frame_feature]
