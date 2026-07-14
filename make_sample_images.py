"""
Generates a few simple synthetic 'street scene' images so the project
has something to annotate and validate out of the box.

NOTE: These are placeholder images for demonstrating the pipeline.
Before pushing to GitHub for a real application, replace these with
actual annotated images (e.g. a handful from a public dataset like
BDD100K or KITTI, annotated yourself in CVAT).
"""
from PIL import Image, ImageDraw
import os

os.makedirs("images", exist_ok=True)

def make_scene(path, boxes, size=(640, 480), bg=(180, 200, 210)):
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)
    # ground
    draw.rectangle([0, size[1] - 120, size[0], size[1]], fill=(90, 90, 90))
    for (x1, y1, x2, y2, color) in boxes:
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=2)
    img.save(path)

# Scene 1: a car and a pedestrian, clean and well-formed
make_scene(
    "images/scene_001.jpg",
    [(80, 300, 220, 400, (200, 50, 50)), (400, 250, 440, 400, (240, 210, 140))],
)

# Scene 2: a truck partially out of frame + a sign
make_scene(
    "images/scene_002.jpg",
    [(500, 260, 700, 420, (60, 90, 160)), (100, 150, 140, 220, (230, 200, 40))],
)

# Scene 3: two overlapping vehicles (used to demonstrate duplicate/overlap detection)
make_scene(
    "images/scene_003.jpg",
    [(150, 280, 320, 400, (80, 150, 80)), (170, 285, 340, 405, (80, 150, 80))],
)

print("Sample images created in images/")
