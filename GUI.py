import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageFilter, ImageEnhance
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
    img = ImageEnhance.Contrast(img).enhance(1.5)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    return img


def segmentation_descriptor(piece_np, num_segments=40):
    img_lab = rgb2lab(piece_np)
    segments = slic(img_lab, n_segments=num_segments, compactness=12)
    seg_ids, counts = np.unique(segments, return_counts=True)

    return {
        "segment_count": int(len(seg_ids)),
        "avg_segment_size": float(np.mean(counts)),
        "segment_mask": segments.tolist()
    }


def split_grid_with_descriptors(img_pil, grid_size, output_folder):
    width, height = img_pil.size
    piece_w = width // grid_size
    piece_h = height // grid_size

    pieces_info = []
    count = 1

    for row in range(grid_size):
        for col in range(grid_size):
            left = col * piece_w
            upper = row * piece_h
            right = left + piece_w
            lower = upper + piece_h

            piece = img_pil.crop((left, upper, right, lower))
            filename = f"piece_{count}.png"
            piece.save(os.path.join(output_folder, filename))

            piece_np = np.array(piece)
            gray = cv2.cvtColor(piece_np, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            contour_points = contours[0].reshape(-1, 2).tolist() if contours else []
            seg_desc = segmentation_descriptor(piece_np)

            pieces_info.append({
                "id": count,
                "filename": filename,
                "bbox": [left, upper, right, lower],
                "size": [piece_w, piece_h],
                "contour": contour_points,
                "segmentation": seg_desc
            })

            count += 1

    with open(os.path.join(output_folder, "pieces_descriptors.json"), "w") as f:
        json.dump(pieces_info, f, indent=2)




def browse_image():
    path = filedialog.askopenfilename(
        filetypes=[("Images", "*.jpg *.png *.jpeg")]
    )
    if path:
        image_path_var.set(path)
        show_preview(path)


def show_preview(path):
    img = Image.open(path)
    img.thumbnail((220, 220))
    photo = ImageTk.PhotoImage(img)
    preview_label.config(image=photo, text="")
    preview_label.image = photo


def run_processing():
    try:
        if not image_path_var.get():
            raise ValueError("Please select an image")

        status_label.config(text="Processing...", fg="#FFA500")
        root.update_idletasks()

        img = enhance_image(load_image(image_path_var.get()))
        grid_size = grid_slider.get()

        output_folder = "output_pieces"
        os.makedirs(output_folder, exist_ok=True)

        split_grid_with_descriptors(img, grid_size, output_folder)

        status_label.config(text="Done Successfully ✔", fg="#00C853")
        messagebox.showinfo("Success", "Puzzle processed successfully!")

    except Exception as e:
        status_label.config(text="Error ❌", fg="red")
        messagebox.showerror("Error", str(e))




root = tk.Tk()
root.title("Puzzle Segmentation Tool")
root.geometry("620x420")
root.configure(bg="#1E1E2F")
root.resizable(False, False)

image_path_var = tk.StringVar()


tk.Label(
    root,
    text="Puzzle Image Segmentation",
    font=("Segoe UI", 18, "bold"),
    bg="#1E1E2F",
    fg="white"
).pack(pady=10)


main_frame = tk.Frame(root, bg="#2B2B3C")
main_frame.pack(padx=15, pady=10, fill="both", expand=True)


left = tk.Frame(main_frame, bg="#2B2B3C")
left.pack(side="left", padx=15, pady=15)

preview_label = tk.Label(
    left,
    text="Image Preview",
    width=28,
    height=14,
    bg="#1E1E2F",
    fg="gray",
    relief="ridge"
)
preview_label.pack(pady=10)

tk.Button(
    left,
    text="📂 Select Image",
    command=browse_image,
    font=("Segoe UI", 10),
    bg="#4CAF50",
    fg="white",
    width=18
).pack(pady=5)


right = tk.Frame(main_frame, bg="#2B2B3C")
right.pack(side="right", padx=15, pady=15, fill="y")

tk.Label(
    right,
    text="Grid Size (NxN)",
    font=("Segoe UI", 12),
    bg="#2B2B3C",
    fg="white"
).pack(pady=10)

grid_slider = tk.Scale(
    right,
    from_=2,
    to=8,
    orient="horizontal",
    length=200,
    bg="#2B2B3C",
    fg="white",
    highlightthickness=0
)
grid_slider.set(2)
grid_slider.pack()

tk.Button(
    right,
    text="🚀 Run Segmentation",
    command=run_processing,
    font=("Segoe UI", 12, "bold"),
    bg="#2196F3",
    fg="white",
    width=20
).pack(pady=30)

status_label = tk.Label(
    right,
    text="Waiting...",
    font=("Segoe UI", 10),
    bg="#2B2B3C",
    fg="gray"
)
status_label.pack()

root.mainloop()