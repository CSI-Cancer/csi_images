# csi_images: image and data utilities for CSI-Cancer

[![PyPI version](https://img.shields.io/pypi/v/csi-images)](https://pypi.org/project/csi-images/)
[![Coverage Badge](docs/theme/coverage.svg)](https://csi-cancer.github.io/csi_images/coverage)

This package is a library for interacting with data, mainly from immunoflourescent 
microscopy, such as cells and the images thereof. 
While much of the functionality is specific to the CSI-Cancer organization, some of the
functionality and structure may be beneficial for the broader community.
Other packages in the CSI-Cancer organization may depend on this package.

Install with:

```commandline
pip install csi_images
```

or

```commandline
pip install csi_images[imageio,rds,dev,all]
```

The base version of the package includes only core data structures and manipulations.

Optional dependencies include:

* `imageio`: for reading and writing images, include .czi files.
* `rds`: for reading and writing RDS files, such as OCULAR outputs.
* `dev`: for development dependencies, including documentation, tests, building, etc.
* `all`: for all optional dependencies.

## Structure

This package contains these modules:

1. `csi_scans.py`: a module for interacting with scan-level files, such as .czi files.
    * `Scan`: a class that contains all of the scan metadata. for interacting with scan
      metadata, such as the slide ID, the path to the scan, and scan parameters.
      Recommend importing via `from csi_images.csi_scans import Scan`
2. `csi_tiles.py`: a module for interacting with tiles, which have a particular (x, y)
   position in the scan. Tiles have several frames taken at the same position.
    * `Tile`: a class for containing a tile's positional data. Imports `csi_scans.py`.
      This class unifies multiple scanners' tile positioning to convert between index
      and (x, y). Recommend importing via `from csi_images.csi_tiles import Tile`
3. `csi_frames.py`: a module for interacting with frames, which are individual images.
   Imports `csi_scans.py` and `csi_tiles.py`. Recommend importing via
   `from csi_images.csi_frames import Frame`
    * `Frame`: a class for containing a frame's metadata. Each frame in a tile has a
      different channel, or light spectrum. The frame **only contains metadata**, but
      enables gathering of the image data through the `get_image()` method. For a list
      of frames, use `get_frames()`. For all frames in a scan, use `get_all_frames()`.
      For many frames, use `[frame.get_image() for frame in frames]`. Recommend
      importing via `from csi_images.csi_frames import Frame`
4. `csi_events.py`: a module for interacting with individual events. Imports
   `csi_scans.py`, `csi_tiles.py`, and `csi_frames.py`.
    * `Event`: a class for containing a single event's metadata and feature data. Key
      metadata (scan, tile, x, y) is required; the others are optional and flexible.
      Contains a convenience function for getting crops of an event, as well as
      functions for determining the position of the event in the slide coordinate frame.
      Recommend importing via `from csi_images.csi_events import Event`
    * `EventArray`: a class for containing a list of events, holding their data in
      pandas dataframes. Contains functions converting back and forth from `Event`s and
      files. Recommend importing via `from csi_images.csi_events import EventArray`

### Planned Features

* `Event.montage()`: Combines crops for an event into side-by-side montages.

## Documentation

For more detailed documentation, check
[the API docs](https://csi-cancer.github.io/csi_images/).

Alternatively, once you have cloned the repository, you
can open up `docs/index.html` in your browser.

To regenerate the documentation, ensure that you
have [installed the package](#installation) and then run:

```commandline
make_docs_for_csi_images
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
