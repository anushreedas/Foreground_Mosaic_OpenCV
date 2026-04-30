# Foreground Mosaic with MOG2 Background Subtraction

A computer vision pipeline that automatically extracts moving foreground objects from video and composites them into a single mosaic image — showing the full trajectory of motion across one frame.

Built for diving competition videos: the pipeline detects each diver at distinct positions in mid-air and places them side-by-side against the static pool background.

---

## How it works

**Step 1 — Build the background model**

The algorithm processes the first 500 frames of the video using [MOG2 Background Subtraction](https://docs.opencv.org/4.x/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html) (Mixture of Gaussians). To handle variable lighting conditions, it selects the brightest frame from every 6-frame window before feeding it into MOG2. The result is a clean background image with no foreground objects.

**Step 2 — Extract the foreground object per position**

For each subsequent 6-frame window, the pipeline:
1. Selects the brightest frame
2. Applies MOG2 to get a foreground mask
3. Refines the mask with **erosion** (removes background noise and water splash artifacts) and **dilation** (expands the region to fully cover the subject including limbs and hair)
4. Applies boundary clipping to suppress false positives from water entry/exit and frame edges
5. Thresholds the result to isolate the largest foreground region

**Step 3 — Compose the mosaic**

Each extracted foreground object is added to the mosaic only if it does not overlap with the previous foreground position — ensuring each appearance in the mosaic shows the subject at a distinct location. The final mosaic is built by horizontally concatenating all position snapshots.

```
Video frames → Brightest frame selection → MOG2 background subtraction
    → Erosion + Dilation → Boundary clipping → Threshold
    → Overlap check → Composite onto background → Mosaic
```

---

## Requirements

```
Python 3.7+
opencv-python
numpy
```

Install dependencies:

```bash
pip install opencv-python numpy
```

---

## Usage

**Single video file:**

```bash
python ForegroundMosaic.py
# Enter filename or directory name: diving_videos/IMG_1794__2175.m4v
```

**Entire directory of videos:**

```bash
python ForegroundMosaic.py
# Enter filename or directory name: diving_videos/
```

Supported formats: `.mp4`, `.avi`, `.mov`, `.m4v`

**Output files** are saved in the same directory as the input video:

- `<videoname>background.jpg` — the extracted background model
- `<videoname>_foregroundmosaic.jpg` — the final foreground mosaic

---

## Implementation details

### Brightest frame selection

Rather than using every frame, the pipeline selects the brightest frame from each 6-frame window. This reduces sensitivity to motion blur during fast movement and handles the flickering lighting conditions common in indoor pool environments.

```python
def findBrightest(imglist):
    greyImages = [cv.cvtColor(f, cv.COLOR_BGR2GRAY) for f in imglist]
    max_idx = 0
    for i in range(len(greyImages)):
        if np.sum(greyImages[i] - greyImages[max_idx]) > 0:
            max_idx = i
    return imglist[max_idx]
```

### Morphological preprocessing

Two rounds of morphological operations are applied with different kernels for different purposes:

- **First erosion** (elliptical, 3×4, 6 iterations): removes small background changes like water ripples
- **Boundary clipping**: zeros out the bottom 350px (water entry/exit), bottom-left 400×200px, and 100px margins on each side
- **Dilation** (elliptical, 100×180, 2 iterations): expands the region to fully encompass the diver's extended limbs
- **Second erosion + dilation** (rectangular kernels): refines the final diver mask for clean compositing

### Overlap detection

Before adding a foreground position to the mosaic, the pipeline checks whether it overlaps with the most recently added position using a bitwise AND of their masks. This prevents duplicate detections of a stationary subject from cluttering the mosaic.

---

## Academic context

Developed for **CSCI-731 Advanced Computer Vision** at Rochester Institute of Technology.

Demonstrates: background modeling, Gaussian mixture models, morphological image processing, video segmentation, and multi-frame compositing using OpenCV.

---

## Author

**Anushree Das**
[LinkedIn](https://linkedin.com/in/anushree-s-das) · [GitHub](https://github.com/anushreedas) · [Medium](https://medium.com/@anushree-das)
