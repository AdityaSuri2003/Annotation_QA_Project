"""
Annotation QA Validator
------------------------
Reads a COCO-style annotation file and checks each bounding box against a
set of quality rules, similar to how an annotation reviewer would check
labeled data before it's approved for use in training.

Checks performed:
  1. Out-of-bounds boxes  - box extends outside the image dimensions
  2. Degenerate boxes     - zero or negative width/height
  3. Likely duplicates    - two boxes of the same category with high overlap (IoU)

Usage:
    python validate_annotations.py annotations/annotations.json
"""
import json
import sys
from itertools import combinations


def to_x1y1x2y2(bbox):
    """Convert COCO [x, y, width, height] to [x1, y1, x2, y2]."""
    x, y, w, h = bbox
    return x, y, x + w, y + h


def iou(box_a, box_b):
    """Compute Intersection-over-Union of two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area

    if union == 0:
        return 0.0
    return inter_area / union


def validate(annotation_path, iou_threshold=0.6):
    with open(annotation_path) as f:
        data = json.load(f)

    images_by_id = {img["id"]: img for img in data["images"]}
    categories_by_id = {c["id"]: c["name"] for c in data["categories"]}

    anns_by_image = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    issues = []  # list of dicts: image, annotation_id(s), issue, detail

    for image_id, anns in anns_by_image.items():
        image = images_by_id.get(image_id)
        if image is None:
            issues.append({
                "image": f"<unknown image_id {image_id}>",
                "annotation_ids": [a["id"] for a in anns],
                "issue": "missing_image_entry",
                "detail": "Annotations reference an image_id not present in 'images'",
            })
            continue

        img_w, img_h = image["width"], image["height"]

        # 1 & 2: bounds and degenerate checks
        for ann in anns:
            x1, y1, x2, y2 = to_x1y1x2y2(ann["bbox"])
            w, h = x2 - x1, y2 - y1

            if w <= 0 or h <= 0:
                issues.append({
                    "image": image["file_name"],
                    "annotation_ids": [ann["id"]],
                    "issue": "degenerate_box",
                    "detail": f"Box has non-positive width/height (w={w}, h={h})",
                })
                continue  # skip further geometric checks on a degenerate box

            if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
                issues.append({
                    "image": image["file_name"],
                    "annotation_ids": [ann["id"]],
                    "issue": "out_of_bounds",
                    "detail": f"Box [{x1},{y1},{x2},{y2}] extends outside image ({img_w}x{img_h})",
                })

        # 3: duplicate / overlap check, same category only
        for ann_a, ann_b in combinations(anns, 2):
            if ann_a["category_id"] != ann_b["category_id"]:
                continue
            box_a = to_x1y1x2y2(ann_a["bbox"])
            box_b = to_x1y1x2y2(ann_b["bbox"])
            score = iou(box_a, box_b)
            if score >= iou_threshold:
                cat_name = categories_by_id.get(ann_a["category_id"], "unknown")
                issues.append({
                    "image": image["file_name"],
                    "annotation_ids": [ann_a["id"], ann_b["id"]],
                    "issue": "likely_duplicate",
                    "detail": f"Two '{cat_name}' boxes overlap with IoU={score:.2f} (threshold {iou_threshold})",
                })

    return issues, anns_by_image, images_by_id


def print_report(issues, anns_by_image, images_by_id):
    total_images = len(images_by_id)
    total_annotations = sum(len(v) for v in anns_by_image.values())
    flagged_images = {i["image"] for i in issues}

    print("=" * 60)
    print("ANNOTATION QA REPORT")
    print("=" * 60)
    print(f"Images checked:      {total_images}")
    print(f"Annotations checked: {total_annotations}")
    print(f"Images with issues:  {len(flagged_images)}")
    print(f"Total issues found:  {len(issues)}")
    print("-" * 60)

    if not issues:
        print("No issues found. All annotations passed QA checks.")
        return

    for issue in issues:
        print(f"[{issue['issue'].upper()}] {issue['image']}  "
              f"(annotation id(s): {issue['annotation_ids']})")
        print(f"    -> {issue['detail']}")
    print("-" * 60)
    print(f"Result: {len(flagged_images)} of {total_images} images need review.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "annotations/annotations.json"
    issues, anns_by_image, images_by_id = validate(path)
    print_report(issues, anns_by_image, images_by_id)
