# Scans & Scanners

Scans are the highest-level data structure, indicating the key parameters of a scan such
as the area scanned, the dimensions of output images, and the channels used. Scans also
include metadata such as the scanner ID, slide ID, where the images should be, etc.

## Coordinate Frames

There are three levels of coordinate frames in a scan. From inside-out, we have:

* **Tile** coordinate frame. Events are provided with simple integer (x, y) pixel
  coordinates, which makes it easy to crop and manipulate images. The origin is at the
  top-left corner as with normal image axes.
* **Scan** coordinate frame. Each scanner has its own coordinate frame, which is
  determined by the scanner's hardware. The scanner coordinate frame is used to convert
  between in-frame pixel coordinates and micrometers. The origin varies by the scanner,
  but generally resides in the top-left of the scanner's movable stage. In cases where
  there are multiple slide slots, the origin is assumed to be at the top left of the
  current slide. The slide may be oriented horizontally, vertically, or upside-down;
  this all depends on the scanner.
* **Slide** coordinate frame. This is a set coordinate frame where:
    * Slide is active area **up**.
    * Slide is oriented **horizontally**.
    * Slide label area is on the **left**.
    * Origin is at the **top-left** corner.

Generally speaking, we should always compare scanners by converting them to the slide
coordinate frame. Events in the scan and slide coordinate frame are referred to in
micrometers ($\mu$m).

![Diagram of the BZScanner coordinate system, which uses a vertical slide alignment with
the label at the bottom. The slide is active area down and tiles zigzag from the
top-left corner across and back. The x-axis points right, the y-axis points
down, and the origin is in the top-left corner. ](bzscanner.png "BZScanner Coordinates")

![Diagram of the Axioscan coordinate system, which uses a horizontal slide alignment
the label on the left. The slide is active area up and tiles go across in row-major
order from the top-left corner. The x-axis points right, and the y-axis points
down, but the origin is in the top-right corner.](axioscan.png "Axioscan Coordinates")

![Diagram of the slide coordinate system, which uses a horizontal slide alignment
the label on the left. The slide is active area up. The x-axis points right, the y-axis
points down, and the origin is in the top-left corner.](slide.png "Slide Coordinates")
