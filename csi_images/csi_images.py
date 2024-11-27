import numpy as np
import pandas as pd
from skimage.measure import regionprops_table


def extract_mask_info(
    mask: np.ndarray,
    images: list[np.ndarray] = None,
    image_labels: list[str] = None,
    properties: list[str] = None,
) -> pd.DataFrame:
    """
    Extracts events from a mask. Originated from @vishnu
    :param mask: mask to extract events from
    :param images: list of intensity images to extract from
    :param image_labels: list of labels for images
    :param properties: list of properties to extract in addition to the defaults:
    label, centroid, axis_major_length. See
    https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
    for additional properties.
    :return: pd.DataFrame with columns: id, x, y, size, or an empty DataFrame
    """
    # Return empty if the mask is empty
    if np.max(mask) == 0:
        return pd.DataFrame()
    # Reshape any intensity images
    if images is not None:
        if isinstance(images, list):
            images = np.stack(images, axis=-1)
        if image_labels is not None and len(image_labels) != images.shape[-1]:
            raise ValueError("Number of image labels must match number of images.")
    # Accumulate any extra properties
    base_properties = ["label", "centroid", "axis_major_length"]
    if properties is not None:
        properties = base_properties + properties
    else:
        properties = base_properties

    # Use skimage.measure.regionprops_table to compute properties
    info = pd.DataFrame(
        regionprops_table(mask, intensity_image=images, properties=properties)
    )

    # Rename columns to match desired output
    info = info.rename(
        columns={
            "label": "id",
            "centroid-0": "y",
            "centroid-1": "x",
            "axis_major_length": "size",
        },
    )
    renamings = {}
    for column in info.columns:
        for i in range(len(image_labels)):
            suffix = f"-{i}"
            if column.endswith(suffix):
                renamings[column] = f"{image_labels[i]}_{column[:-len(suffix)]}"
    info = info.rename(columns=renamings)

    return info


def make_rgb(
    images: list[np.ndarray], colors=list[tuple[float, float, float]]
) -> np.ndarray:
    """
    Combine multiple channels into a single RGB image.
    :param images: list of numpy arrays representing the channels.
    :param colors: list of RGB tuples for each channel.
    :return:
    """
    if len(images) == 0:
        raise ValueError("No images provided.")
    if len(colors) == 0:
        raise ValueError("No colors provided.")
    if len(images) != len(colors):
        raise ValueError("Number of images and colors must match.")
    if not all([isinstance(image, np.ndarray) for image in images]):
        raise ValueError("Images must be numpy arrays.")
    if not all([len(c) == 3 for c in colors]):
        raise ValueError("Colors must be RGB tuples.")

    # Create an output with same shape and larger type to avoid overflow
    dims = images[0].shape
    dtype = images[0].dtype
    if dtype not in [np.uint8, np.uint16]:
        raise ValueError("Image dtype must be uint8 or uint16.")
    rgb = np.zeros((*dims, 3), dtype=np.uint16 if dtype == np.uint8 else np.uint32)

    # Combine images with colors (can also be thought of as gains)
    for image, color in zip(images, colors):
        if image.shape != dims:
            raise ValueError("All images must have the same shape.")
        if image.dtype != dtype:
            raise ValueError("All images must have the same dtype.")
        rgb[..., 0] += (image * color[0]).astype(rgb.dtype)
        rgb[..., 1] += (image * color[1]).astype(rgb.dtype)
        rgb[..., 2] += (image * color[2]).astype(rgb.dtype)

    # Cut off any overflow and convert back to original dtype
    rgb = np.clip(rgb, np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)
    return rgb
