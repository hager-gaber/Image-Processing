from PIL import Image, ImageFilter, ImageEnhance
import os
import json
import cv2
import numpy as np

from skimage.segmentation import slic
from skimage.color import rgb2lab


def load_image(path):
    img = Image.open(path)
    return img.convert("RGB")


def enhance_image(img):
    img = img.filter(ImageFilter.MedianFilter(3))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(2.0)
    return img


def create_output_folder(folder_name):
    os.makedirs(folder_name, exist_ok=True)
    return folder_name



def segmentation_descriptor(piece_np, num_segments=40):
    """Extract segmentation features using SLIC (no ML)."""
    img_lab = rgb2lab(piece_np)

    segments = slic(img_lab, n_segments=num_segments, compactness=12)

    
    seg_ids, counts = np.unique(segments, return_counts=True)
    segment_count = int(len(seg_ids))
    avg_segment_size = float(np.mean(counts))

    return {
        "segment_count": segment_count,
        "avg_segment_size": avg_segment_size,
        "segment_mask": segments.tolist()
    }


def split_grid_with_descriptors(img_pil, grid_size, output_folder):
    img_np = np.array(img_pil)
    width, height = img_pil.size
    piece_w = width // grid_size
    piece_h = height // grid_size
    count = 1
    pieces_info = []

    for row in range(grid_size):
        for col in range(grid_size):
            left = col * piece_w
            upper = row * piece_h
            right = left + piece_w
            lower = upper + piece_h

            piece = img_pil.crop((left, upper, right, lower))
            filename = f"piece_{count}.png"
            piece.save(f"{output_folder}/{filename}")

            piece_np = np.array(piece)
            gray = cv2.cvtColor(piece_np, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_points = contours[0].reshape(-1, 2).tolist() if contours else []

            
            seg_desc = segmentation_descriptor(piece_np)

            piece_dict = {
                "id": count,
                "filename": filename,
                "size": [piece_w, piece_h],
                "bbox": [left, upper, right, lower],
                "contour": contour_points,
                "segmentation": seg_desc  
            }

            pieces_info.append(piece_dict)
            count += 1

    json_path = os.path.join(output_folder, "pieces_descriptors.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(pieces_info, f, indent=2)


def main():
    image_path = "91.jpg"
    grid_size = 4
    img = load_image("91.jpg")
    img = enhance_image(img)
    folder = create_output_folder("output_pieces")
    split_grid_with_descriptors(img, grid_size, folder)
    print(f"Done! Puzzle pieces saved in '{folder}' with descriptors.")


main()
