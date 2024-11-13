import numpy as np


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
