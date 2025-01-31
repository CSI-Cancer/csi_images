# csi_images: image and data utilities for CSI-Cancer

[![PyPI version](https://img.shields.io/pypi/v/csi-images)](https://pypi.org/project/csi-images/)
![Coverage Badge](docs/coverage.svg)
![Tests Badge](docs/tests.svg)

This package is a library for interacting with data, mainly from immunoflourescent
microscopy, such as cells and the images thereof.
While much of the functionality is specific to the CSI-Cancer organization, some of the
functionality and structure may be beneficial for the broader community.
Other packages in the CSI-Cancer organization may depend on this package.

Install with:

```bash
pip install csi_images
```

or

```bash
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
    * `Tile`: a class for containing a tile's positional data. Includes a reference to a
      `Scan`. This class unifies multiple scanners' tile positioning to convert between
      index and (x, y). Recommend importing via `from csi_images.csi_tiles import Tile`
3. `csi_frames.py`: a module for interacting with frames, which are individual images.
   Imports `csi_scans.py` and `csi_tiles.py`. Recommend importing via
   `from csi_images.csi_frames import Frame`
    * `Frame`: a class for containing a frame's metadata. Includes a reference to a
      `Tile`. For a list of frames for a tile, use `get_frames(tile)`. For
      all frames in a scan, use `get_all_frames(scan)`. Each frame in a tile has a
      different channel, or light spectrum. The frame **only contains metadata**,
      but enables gathering of the image data through the `get_image()` method. For
      many frames, use `[frame.get_image() for frame in frames]`. Recommend importing
      via `from csi_images.csi_frames import Frame`
4. `csi_events.py`: a module for interacting with individual events. Imports
   `csi_scans.py`, `csi_tiles.py`, and `csi_frames.py`.
    * `Event`: a class for containing a single event's metadata and feature data. Key
      metadata (scan, tile, x, y) is required; the others are optional and flexible.
      Contains functions for determining event position in different coordinate frames
      and performing crop and montage image manipulations, as well as name conventions
      for saving crops and images. Recommend importing via
      `from csi_images.csi_events import Event`
    * `EventArray`: a class for containing a list of events, holding their data in
      Pandas DataFrames. Contains functions converting back and forth from `Event`s and
      files. Recommend importing via `from csi_images.csi_events import EventArray`
5. `csi_images.py`: a module for image manipulation functions. These functions are
   developed to be portable and contain all of the image-related dependencies, making it
   possible to use the other modules without image-related dependencies. Recommend
   importing via `from csi_images import csi_images`

## Documentation

For more detailed documentation, check
[the API docs](https://csi-cancer.github.io/csi_images/).

Alternatively, once you have cloned the repository, you can open up `docs/index.html` in
your browser.

To regenerate the documentation, ensure that you have installed the package with
development dependencies and then run:

```bash
docs/make_docs.sh
```

## Development Installation

1. Activate your `conda` (`conda activate yourenv`) or
   `venv` (`source path/to/your/venv/bin/activate`) environment first.
2. Clone `csi_images` and install:

```bash
cd ~/path/to/your/repositories
git clone git@github.com:CSI-Cancer/csi_images.git
pip install ./csi_images
```

Alternatively, you can "editable" install the package, which will allow you to make
changes to the package and have them reflected in your environment without reinstalling:

```bash
pip install -e ./csi_images
```

This will add symbolic links to your `site-packages` directory instead of copying the
package files over.
