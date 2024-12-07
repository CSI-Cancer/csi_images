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


def make_montage(
    images: list[np.ndarray],
    order: list[int] = None,
    composites: dict[int, tuple[float, float, float]] = None,
    border_size: int = 2,
    horizontal: bool = True,
    dtype=np.uint8,
) -> np.ndarray:
    """
    Combine multiple images into a single montage based on order.
    Can include a composite (always first).
    :param images: list of numpy arrays representing the images.
    :param order: list of indices for the images going into the montage.
    :param composites: dictionary of indices and RGB tuples for a composite.
    :param border_size: width of the border between images.
    :param horizontal: whether to stack images horizontally or vertically.
    :param dtype: the dtype of the output montage.
    :return: numpy array representing the montage.
    """
    if len(images) == 0:
        raise ValueError("No images provided.")
    if not all([isinstance(image, np.ndarray) for image in images]):
        raise ValueError("Images must be numpy arrays.")
    if not all([len(image.shape) == 2 for image in images]):
        raise ValueError("Images must be 2D.")
    if composites is not None and not all([len(c) == 3 for c in composites.values()]):
        raise ValueError("Composites must be RGB tuples.")
    if order is None and composites is None:
        raise ValueError("No images or composites requested.")

    # Populate the montage with black
    n_images = len(order) if order is not None else 0
    n_images += 1 if composites is not None else 0
    montage = np.zeros(
        get_montage_size(images[0].shape, n_images, border_size, horizontal),
        dtype=dtype,
    )

    # Populate the montage with images
    offset = border_size  # Keeps track of the offset for the next image
    image_height, image_width = images[0].shape

    # Composite first
    if composites is not None:
        image = make_rgb(
            [images[i] for i in composites.keys()],
            list(composites.values()),
        )
        image = scale_bit_depth(image, dtype)
        if horizontal:
            montage[
                border_size : border_size + image_height,
                offset : offset + image_width,
            ] = image
            offset += image_width + border_size
        else:
            montage[
                offset : offset + image_height,
                border_size : border_size + image_width,
            ] = image
            offset += image_height + border_size

    # Grayscale order next
    for i in order:
        image = images[i]
        image = scale_bit_depth(image, dtype)
        image = np.tile(image[..., None], (1, 1, 3))  # Make 3-channel
        if horizontal:
            montage[
                border_size : border_size + image_height,
                offset : offset + image_width,
            ] = image
            offset += image_width + border_size
        else:
            montage[
                offset : offset + image_height,
                border_size : border_size + image_width,
            ] = image
            offset += image_height + border_size

    return montage


def get_montage_size(
    image_shape: tuple[int, int],
    n_images: int,
    border_size: int = 2,
    horizontal: bool = True,
) -> tuple[int, int, int]:
    """
    Determine the size of the montage based on the images and order.
    :param image_shape: tuple of height, width of the base images going into the montage.
    :param n_images: how many images are going into the montage, including composite.
    :param border_size: width of the border between images.
    :param horizontal: whether to stack images horizontally or vertically.
    :return: tuple of the height, width, and channels (always 3) of the montage.
    """
    if len(image_shape) != 2:
        raise ValueError("Image shape must be a tuple of height, width.")
    if image_shape[0] < 1 or image_shape[1] < 1:
        raise ValueError("Image shape must be positive.")
    if not isinstance(n_images, int) or n_images < 1:
        raise ValueError("Number of images must be a positive integer.")

    # Determine the size of the montage
    if horizontal:
        n_rows = 1
        n_cols = n_images
    else:
        n_rows = n_images
        n_cols = 1

    # Determine the montage size
    image_height, image_width = image_shape
    montage_height = n_rows * image_height + (n_rows + 1) * border_size
    montage_width = n_cols * image_width + (n_cols + 1) * border_size

    return montage_height, montage_width, 3  # 3 for RGB


def scale_bit_depth(
    image: np.ndarray, dtype: np.dtype, real_bits: int = None
) -> np.ndarray:
    """
    Converts the image to the desired bit depth, factoring in real bit depth.
    :param image: numpy array representing the image.
    :param dtype: the desired dtype of the image.
    :param real_bits: the actual bit depth of the image, such as from a 14-bit camera.
    :return: numpy array representing the image with the new dtype.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a numpy array.")
    if not np.issubdtype(image.dtype, np.unsignedinteger) and not np.issubdtype(
        image.dtype, np.floating
    ):
        raise ValueError("Input image dtype must be an unsigned integer or float.")
    if np.issubdtype(image.dtype, np.floating) and (
        np.min(image) < 0 or np.max(image) > 1
    ):
        raise ValueError("Image values must be between 0 and 1.")
    if not np.issubdtype(dtype, np.unsignedinteger) and not np.issubdtype(
        dtype, np.floating
    ):
        raise ValueError("Output dtype must be an unsigned integer or float.")

    # First, determine the scaling required for the real bit depth
    scale = 1
    if real_bits is not None and np.issubdtype(image.dtype, np.unsignedinteger):
        dtype_bit_depth = np.iinfo(image.dtype).bits
        if real_bits > dtype_bit_depth:
            raise ValueError("Real bits must be less than or equal to image bit depth")
        elif real_bits < dtype_bit_depth:
            # We should scale up the values to the new bit depth
            if np.max(image) > 2**real_bits:
                raise ValueError("Image values exceed real bit depth; already scaled?")
            scale = np.iinfo(image.dtype).max / (2**real_bits - 1)

    # Already validated that the min is 0; determine the max
    if np.issubdtype(image.dtype, np.unsignedinteger):
        in_max = np.iinfo(image.dtype).max
    else:
        in_max = 1.0
    if np.issubdtype(dtype, np.unsignedinteger):
        out_max = np.iinfo(dtype).max
    else:
        out_max = 1.0

    # Scale the image to the new bit depth
    scale = scale * out_max / in_max
    image = (image * scale).astype(dtype)
    return image
