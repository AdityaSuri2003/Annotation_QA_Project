# Annotation QA Validator

A small project that reviews bounding-box annotations for quality issues,
the same way an annotation reviewer checks labeled data before it's approved
for use in training a model. Built while applying for a Data Annotator role
to demonstrate the review/validation side of annotation work, not just
drawing boxes.

## What it does

Given a set of images and their bounding-box annotations (COCO format), the
validator checks every box against three rules:

1. **Out-of-bounds** - the box extends past the edges of the image
2. **Degenerate boxes** - zero or negative width/height (a box that collapsed
   to a line or a point)
3. **Likely duplicates** - two boxes of the same category overlapping heavily
   (IoU above a threshold), which usually means the same object got labeled
   twice

Flagged boxes are drawn in red on the output images, clean boxes in green, so
you can see at a glance which images need a second look.

## Project structure

```
annotation-qa-project/
├── images/                  sample images
├── annotations/
│   └── annotations.json     bounding-box annotations (COCO-style)
├── output/                  generated review images (created by visualize.py)
├── validate_annotations.py  runs the QA checks, prints a report
├── visualize.py             draws flagged/clean boxes back onto the images
├── make_sample_images.py    generates the placeholder sample images
└── requirements.txt
```

> The images included here are simple generated placeholders so the project
> runs end to end out of the box. In a real annotation batch, `images/` and
> `annotations/annotations.json` would come from actually labeling images in
> a tool like CVAT.

## Running it

```bash
pip install -r requirements.txt

# run the QA checks and print a report
python validate_annotations.py annotations/annotations.json

# draw the flagged/clean boxes onto the images
python visualize.py annotations/annotations.json
```

## Example output

```
============================================================
ANNOTATION QA REPORT
============================================================
Images checked:      3
Annotations checked: 9
Images with issues:  2
Total issues found:  5
------------------------------------------------------------
[OUT_OF_BOUNDS] scene_002.jpg  (annotation id(s): [3])
    -> Box [500,260,700,420] extends outside image (640x480)
[LIKELY_DUPLICATE] scene_003.jpg  (annotation id(s): [6, 7])
    -> Two 'car' boxes overlap with IoU=0.73 (threshold 0.6)
------------------------------------------------------------
Result: 2 of 3 images need review.
```

## Why these three checks

They're the errors that come up constantly in real annotation batches:
a box drawn slightly past the image edge, a leftover duplicate box from
copy-pasting a label, or a box that got dragged to nothing by accident.
Catching them programmatically is exactly the kind of consistency check
an annotation QA pipeline relies on before data gets used for training.

## Possible extensions

- Add a check for missing annotations (an image with zero boxes)
- Support Pascal VOC XML in addition to COCO JSON
- Add a confidence/consistency check if annotating with multiple people
