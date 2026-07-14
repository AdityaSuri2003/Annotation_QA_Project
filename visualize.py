"""
Draws bounding boxes from the annotation file onto their images, color-coding
flagged boxes (red) vs. clean boxes (green), and saves the result to output/.

Usage:
    python visualize.py annotations/annotations.json
"""
import json
import os
import sys
from PIL import Image, ImageDraw, ImageFont
from validate_annotations import validate, to_x1y1x2y2


def flagged_annotation_ids(issues):
    flagged = set()
    for issue in issues:
        flagged.update(issue["annotation_ids"])
    return flagged


def draw_boxes(annotation_path, images_dir="images", out_dir="output"):
    os.makedirs(out_dir, exist_ok=True)
    issues, anns_by_image, images_by_id = validate(annotation_path)
    flagged_ids = flagged_annotation_ids(issues)

    with open(annotation_path) as f:
        data = json.load(f)
    categories_by_id = {c["id"]: c["name"] for c in data["categories"]}

    for image_id, anns in anns_by_image.items():
        image_meta = images_by_id.get(image_id)
        if image_meta is None:
            continue

        img_path = os.path.join(images_dir, image_meta["file_name"])
        if not os.path.exists(img_path):
            print(f"Skipping {img_path} (not found)")
            continue

        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        for ann in anns:
            x1, y1, x2, y2 = to_x1y1x2y2(ann["bbox"])
            is_flagged = ann["id"] in flagged_ids
            color = (220, 30, 30) if is_flagged else (30, 180, 60)
            label = categories_by_id.get(ann["category_id"], "?")
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1 + 2, max(0, y1 - 14)), label, fill=color)

        out_path = os.path.join(out_dir, f"reviewed_{image_meta['file_name']}")
        img.save(out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "annotations/annotations.json"
    draw_boxes(path)
