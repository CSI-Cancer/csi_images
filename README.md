# [csi_images](https://pypi.org/project/csi-images/)

This package contains classes and functions for interacting with images and image data.
While much of the functionality is specific to the CSI-Cancer organization, some of the
functionality and structure may be beneficial for the broader community.
Other packages in the CSI-Cancer organization may depend on this package.

Install with: 
```
pip install csi_images
```

## Structure

This package contains these helpful modules:

1. `csi_scans.py`: a module for interacting with scan-level files, such as .czi files.
    * `Scan`: a class that contains all of the scan metadata. for interacting with scan
      metadata, such as the slide ID, the path to the scan, and scan parameters.
2. `csi_tiles.py`: a module for interacting with tiles, which have a particular (x, y)
   position in the scan. Tiles have several frames taken at the same position.
    * `Tile`: a class for containing a tile's metadata data. Imports `csi_scans.py`.
        * `get_tiles()`: gets the tile of a scan at a particular coordinate.
        * `get_all_tiles()`: gets all of the tiles of a scan.
        * `get_tiles_by_row_col()`: gets the tiles in particular rows and columns.
        * `get_tiles_by_xy_bounds()`: gets the tiles within particular x and y bounds.
5. `csi_frames.py`: a module for interacting with frames, which are individual images
   from the scan. Each frame in a tile has a different channel, or light spectrum.
   Imports `csi_scans.py` and `csi_tiles.py`. 
    * `Frame`: a class for containing a frame's metadata.
        * `get_image()`: gets the image of the frame.
        * `get_frames()`: gets the frames of a tile.
        * `get_all_frames()`: gets all of the frames of a scan.
        * `make_rgb_image()`: creates an RGB image from the frames.
6. `csi_events.py`: a module for interacting with individual events. Imports
   `csi_scans.py`, `csi_tiles.py`, and `csi_frames.py`.
    * `Event`: a class for containing an event's metadata and feature data. Key metadata
      (scan, tile, x, y) is required; the others are optional and flexible.
        * `get_scan_position()`: gets the x, y position of the event in the scan
          coordinate system.
        * `get_slide_position()`: gets the x, y position of the event in the slide
          coordinate system.
        * `extract_event_images()`: extracts the images of the event from the scan,
          reading from the scan images to do so. Convenient for getting a few events.
        * `crop_images_to_event()`: crops the images of the event from the scan, using
          the passed-in images. More efficient when getting many events.
    * `extract_all_event_images()`: efficiently extracts the images of a list of events.
    * `get_features_as_dataframe()`: combines the features from a list of events into a
      DataFrame.
    * `get_metadata_as_dataframe()`: combines the metadata from a list of events into a
      DataFrame.
    * `save_to_hdf5()`: loads a list of events from a CSV file. **Not implemented yet.**
    * `load_from_hdf5()`: montages a list of events. **Not implemented yet.**
    * `save_to_csv()`: loads a list of events from a CSV file. **Not implemented yet.**
    * `load_from_csv()`: montages a list of events. **Not implemented yet.**

### Planned Features

* `csi_events.py`: a module for interacting with individual events. Imports
  `csi_scans.py`, `csi_tiles.py`, and `csi_frames.py`.
    * `montage_events()`: Combines crops for an event into side-by-side montages.

## Documentation

For more detailed documentation, check 
[the API docs](https://csi-cancer.github.io/csi_images/).

Alternatively, once you have cloned the repository, you 
can open up `docs/index.html` in your browser.

To regenerate the documentation, ensure that you
have [installed the package](#installation) and then run:

```commandline
make_docs_csi_images
```

## Development Installation

1. Activate your `conda` (`conda activate yourenv`) or
   `venv` (`source path/to/your/venv/bin/activate`) environment first.
2. Clone `csi_images` and install:

```commandline
cd ~/path/to/your/repositories
git clone git@github.com:CSI-Cancer/csi_images.git
pip install ./csi_images
```

Alternatively, you can "editable" install the package, which will allow you to make
changes to the package and have them reflected in your environment without reinstalling:

```commandline
pip install -e ./csi_images
```

This will add symbolic links to your `site-packages` directory instead of copying the
package files over.
