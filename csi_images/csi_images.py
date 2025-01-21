import warnings
from typing import Literal

import cv2
import numpy as np
import pandas as pd

from PIL import Image, ImageFont, ImageDraw
from skimage.measure import regionprops_table

# Avoid opening multiple fonts and re-opening fonts
opened_font: ImageFont.FreeTypeFont | None = None


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
    base_properties = ["label", "centroid"]
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
        },
    )
    # Also rename channel-specific columns if necessary
    renamings = {}
    for column in info.columns:
        if column.find("-") != -1:
            for i in range(len(image_labels)):
                suffix = f"-{i}"
                if column.endswith(suffix):
                    renamings[column] = f"{image_labels[i]}_{column[:-len(suffix)]}"
    info = info.rename(columns=renamings)

    return info


def add_mask_overlay(
    images: np.ndarray | list[np.ndarray],
    mask: np.ndarray[np.uint8],
    overlay_color: tuple[float, float, float] = (0.8, 1, 0),
):
    """
    Creates a 1-pixel wide border around the mask in the image.
    :param images: (H, W, 3), or (H, W) image or list of images.
    :param mask: (H, W) binary mask to overlay on the image.
    :param overlay_color: color of the outline for RGB images.
    Ignored for grayscale images.
    :return:
    """
    results = []
    # Temporarily make into a list
    return_array = False
    if isinstance(images, np.ndarray):
        images = [images]
        return_array = True

    # Get the mask outline
    mask_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    outline = cv2.morphologyEx(mask, cv2.MORPH_DILATE, mask_kernel) - mask

    # Add mask overlay to images
    for image in images:
        if image.shape[:2] != mask.shape:
            raise ValueError("Image and mask must have the same shape.")
        if np.issubdtype(image.dtype, np.unsignedinteger):
            # Unsigned integer; scale outline to image range
            this_outline = outline.astype(image.dtype) * np.iinfo(image.dtype).max
        else:
            # Floating point; scale outline to [0, 1]
            this_outline = outline.astype(image.dtype)
        if len(image.shape) == 3:
            # (H, W, 3) RGB image
            # Scale outline with color and match dtype
            this_outline = np.stack([this_outline * c for c in overlay_color], axis=-1)
            # Set outline to 0, then to outline color
            result = image * np.stack([1 - outline] * 3, axis=-1)
            result += this_outline.astype(image.dtype)
        elif len(image.shape) == 2:
            # (H, W) grayscale image
            result = image * (1 - outline)
            result += this_outline.astype(image.dtype)
        results.append(result)

    if return_array:
        return results[0]
    else:
        return results


def make_rgb(
    images: list[np.ndarray],
    colors=list[tuple[float, float, float]],
    mask: np.ndarray[np.uint8] = None,
    mask_mode: Literal["overlay", "hard"] = "overlay",
    overlay_color: tuple[float, float, float] = (0.8, 1, 0),
) -> np.ndarray:
    """
    Combine multiple channels into a single RGB image.
    :param images: list of numpy arrays representing the channels.
    :param colors: list of RGB tuples for each channel.
    :param mask: numpy array representing a mask to overlay on the image.
    :param mask_mode: whether to overlay the mask or use it as a hard mask.
    :param overlay_color: color of the outline for RGB images.
    Ignored for grayscale images.
    :return: (H, W, 3) numpy array representing the RGB image.
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

    if dtype == np.uint8:
        temp_dtype = np.uint16
    elif dtype == np.uint16:
        temp_dtype = np.uint32
    else:
        temp_dtype = np.float64
    rgb = np.zeros((*dims, 3), dtype=temp_dtype)

    # Combine images with colors (can also be thought of as gains)
    for image, color in zip(images, colors):
        if image.shape != dims:
            raise ValueError("All images must have the same shape.")
        if image.dtype != dtype:
            raise ValueError("All images must have the same dtype.")
        rgb += np.stack([image * c for c in color], axis=-1).astype(temp_dtype)

    # Cut off any overflow and convert back to original dtype
    if np.issubdtype(dtype, np.unsignedinteger):
        max_value = np.iinfo(dtype).max
    else:
        max_value = 1.0
    rgb = np.clip(rgb, 0, max_value).astype(dtype)

    # Add a mask if desired
    if mask is not None:
        if mask.shape != dims:
            raise ValueError("Mask must have the same shape as the images.")
        if mask_mode == "overlay":
            rgb = add_mask_overlay(rgb, mask, overlay_color)
        elif mask_mode == "hard":
            rgb = rgb * np.stack([mask] * 3, axis=-1)
        else:
            raise ValueError("Mask mode must be 'overlay' or 'hard'.")

    return rgb


def make_montage(
    images: list[np.ndarray],
    order: list[int] | None,
    composites: dict[int, tuple[float, float, float]] | None,
    mask: np.ndarray[np.uint8] = None,
    labels: list[str] = None,
    mask_mode: Literal["overlay", "hard"] = "overlay",
    overlay_color: tuple[float, float, float] = (0.8, 1, 0),
    label_font: str = "Roboto-Regular.ttf",
    label_size: int | float = 0.18,
    label_outline: bool = True,
    colored_labels: bool = True,
    border_size: int = 1,
    horizontal: bool = True,
    dtype=np.uint8,
) -> np.ndarray:
    """
    Combine multiple images into a single montage based on order.
    Can include a composite (always first).
    :param images: list of numpy arrays representing the images.
    :param order: list of indices for the images going into the montage or None.
    :param composites: dictionary of indices and RGB tuples for a composite or None.
    :param mask: numpy array representing a mask to overlay on the image.
    :param mask_mode: whether to overlay the mask or use it as a hard mask.
    :param overlay_color: color of the outline for RGB images. Ignored for grayscale.
    :param labels: list of labels for the images. If length == len(order), will apply to
    grayscale images only; if length == len(order) + 1 and composites exist, will apply
    to all images.
    :param label_font: path to a font file for labels. See PIL.ImageFont for details.
    :param label_size: size of the font for labels. If a float, calculates a font size
    as a fraction of the image size.
    :param label_outline: whether to draw an outline around the label text.
    :param colored_labels: whether to color the labels based on the composites.
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

    n_images = len(order) if order is not None else 0
    n_images += 1 if composites is not None else 0

    if n_images == 0:
        raise ValueError("No images or composites requested.")

    # Adapt label font size if necessary
    if isinstance(label_size, float):
        label_size = int(images[0].shape[1] * label_size)

    # Populate the montage with black
    montage = np.full(
        get_montage_shape(images[0].shape, n_images, border_size, horizontal),
        np.iinfo(dtype).max,  # White fill
        dtype=dtype,
    )

    # Load font if necessary
    global opened_font
    if labels is not None and len(order) <= len(labels) <= n_images:
        if (
            opened_font is None
            or opened_font.path != label_font
            or opened_font.size != label_size
        ):
            try:
                opened_font = ImageFont.truetype(label_font, label_size)
            except OSError:
                warnings.warn(f"Could not load font {label_font}. Using defaults.")
                opened_font = ImageFont.load_default(label_size)
    elif labels is not None:
        raise ValueError("Number of labels must be 0, match order, or match images.")

    # Populate the montage with images
    offset = border_size  # Keeps track of the offset for the next image
    image_height, image_width = images[0].shape

    # Composite first
    if composites is not None and len(composites) > 0:
        image = make_rgb(
            [images[i] for i in composites.keys()],
            list(composites.values()),
            mask,
            mask_mode,
            overlay_color,
        )

        if labels is not None and len(labels) == n_images:
            image = scale_bit_depth(image, np.uint8)  # Required for PIL
            # Draw a label on the composite
            pillow_image = Image.fromarray(image)
            # Determine the fill color based on the average intensity of the image
            included_height = max(label_size * 2, image.shape[1])
            if get_image_lightness(image[:, -included_height:, :]) > 50:
                text_fill = (0, 0, 0)
                outline_fill = (255, 255, 255)
            else:
                text_fill = (255, 255, 255)
                outline_fill = (0, 0, 0)
            draw = ImageDraw.Draw(pillow_image, "RGB")
            draw.text(
                (image.shape[0] // 2, image.shape[1]),
                labels[0],
                fill=text_fill,
                anchor="md",  # Middle, descender (absolute bottom of font)
                font=opened_font,
                stroke_width=round(label_size / 10) if label_outline else 0,
                stroke_fill=outline_fill,
            )
            image = np.asarray(pillow_image)
            labels = labels[1:]

        # Scale to desired dtype
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
    order = [] if order is None else order
    for i, o in enumerate(order):
        image = images[o]
        image = np.tile(image[..., None], (1, 1, 3))  # Make 3-channel

        if mask is not None:
            if mask_mode == "overlay":
                image = add_mask_overlay(image, mask, overlay_color)
            elif mask_mode == "hard":
                image *= np.stack([mask] * 3, axis=-1)
            else:
                raise ValueError("Mask mode must be 'overlay' or 'hard'.")

        if labels is not None and len(labels) == len(order):
            image = scale_bit_depth(image, np.uint8)  # Required for PIL
            pillow_image = Image.fromarray(image)
            if colored_labels and o in composites:
                text_fill = tuple(round(255 * rgb_f) for rgb_f in composites[o])
                if get_lightness(composites[o]) > 50:
                    outline_fill = (0, 0, 0)
                else:
                    outline_fill = (255, 255, 255)
            else:
                # Determine the color based on the average intensity of the image
                included_height = max(label_size * 2, image.shape[1])
                if get_image_lightness(image[:, -included_height:, :]) > 50:
                    text_fill = (0, 0, 0)
                    outline_fill = (255, 255, 255)
                else:
                    text_fill = (255, 255, 255)
                    outline_fill = (0, 0, 0)
            draw = ImageDraw.Draw(pillow_image, "RGB")
            draw.text(
                (image.shape[0] // 2, image.shape[1]),
                labels[i],
                fill=text_fill,
                anchor="md",  # Middle, descender (absolute bottom of font)
                font=opened_font,
                stroke_width=round(label_size / 10) if label_outline else 0,
                stroke_fill=outline_fill,
            )
            image = np.asarray(pillow_image)

        # Scale to desired dtype
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

    return montage


def get_montage_shape(
    image_shape: tuple[int, int],
    n_images: int,
    border_size: int = 1,
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


def get_image_lightness(image: np.ndarray) -> float:
    """
    Calculate the lightness of an sRGB image, taking shortcuts for speed.
    :param image: numpy array representing the sRGB image.
    :return: approximate perceived lightness of the image, from 0 to 100.
    """
    # Scale image to [0, 1]
    if np.issubdtype(image.dtype, np.unsignedinteger):
        image = image / np.iinfo(image.dtype).max
    # Rough conversion to linear RGB
    image = image**2.2
    # Average to a single color and return that color's lightness
    color = np.mean(image, axis=(0, 1))
    return get_lightness((color[0], color[1], color[2]), srgb=False)


def get_lightness(color: tuple[float, float, float], srgb: bool = True) -> float:
    """
    Calculate the lightness of an sRGB color, taking shortcuts for speed.
    :param color: an sRGB or linear RGB color as a tuple, with values in [0, 1].
    :param srgb: whether the color is in sRGB or linear RGB.
    :return: approximate perceived lightness of the color, from 0 to 100.
    """
    if srgb:
        # Convert to linear color, rough and quick
        rgb = color[0] ** 2.2, color[1] ** 2.2, color[2] ** 2.2
    else:
        rgb = color
    # Convert to luminance
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    # Convert to perceived lightness
    if luminance <= 0.008856:
        return 903.3 * luminance
    else:
        return 116 * luminance ** (1 / 3) - 16
