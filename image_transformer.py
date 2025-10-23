import numpy as np
import sys
from PIL import Image
import cv2

def load_and_resize_image(path, size):
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size), Image.Resampling.LANCZOS)  # resize keeping aspect ratio
    
    # After thumbnail, pad image to exactly (size, size)
    new_img = Image.new("RGB", (size, size), (0,0,0))
    offset_x = (size - img.width) // 2
    offset_y = (size - img.height) // 2
    new_img.paste(img, (offset_x, offset_y))
    
    return np.array(new_img)

def fast_pixel_mapping(source_img, target_img):
    h, w, _ = source_img.shape
    num_pixels = h * w

    source_flat = source_img.reshape((num_pixels, 3))
    target_flat = target_img.reshape((num_pixels, 3))

    source_order = np.argsort(np.sum(source_flat, axis=1))
    target_order = np.argsort(np.sum(target_flat, axis=1))

    pixel_positions = np.zeros(num_pixels, dtype=int)
    pixel_positions[target_order] = source_order

    return pixel_positions.reshape((h, w)), source_flat

def create_transition_video(mapping, source_pixels, shape, output="transition.mp4", steps=30, fps=30, hold_duration_sec=2):
    h, w = shape
    num_pixels = h * w

    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    coords = np.stack((ys, xs), axis=-1).reshape((num_pixels, 2))

    start_coords = coords
    end_coords = coords[np.argsort(mapping.reshape(-1))]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output, fourcc, fps, (w, h))

    hold_frames = int(hold_duration_sec * fps)

    for step in range(steps):
        alpha = step / (steps - 1)
        interp_coords = (1 - alpha) * start_coords + alpha * end_coords
        interp_coords_rounded = np.round(interp_coords).astype(int)
        interp_coords_rounded = np.clip(interp_coords_rounded, 0, [h - 1, w - 1])

        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[interp_coords_rounded[:, 0], interp_coords_rounded[:, 1]] = source_pixels

        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Hold first and last image
        if step == steps - 1 or step == 0:
            for _ in range(hold_frames):
                video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # Hold the final frame for `hold_duration_sec` seconds
    

    video.release()

def main():
    img_1_path = sys.argv[1]
    img_2_path = sys.argv[2]
    output_video_path = sys.argv[3]

    MAX_SIZE = 512  # max dimension to resize to
    print("Loading and resizing images...", end="\r")
    source_img = load_and_resize_image(img_1_path, MAX_SIZE)
    target_img = load_and_resize_image(img_2_path, MAX_SIZE)

    print("Computing pixel mapping...", end="\r")
    mapping, source_pixels = fast_pixel_mapping(source_img, target_img)
    print("Creating transition video...", end="\r")
    create_transition_video(mapping, source_pixels, source_img.shape[:2], output=output_video_path, steps=300, fps=30, hold_duration_sec=2)
    print(f"Video saved to {output_video_path}", end="\n")

if __name__ == "__main__":
    main()

