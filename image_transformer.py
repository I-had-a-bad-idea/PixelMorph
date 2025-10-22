import numpy as np
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

def create_transition_video(mapping, source_pixels, shape, output="transition.mp4", steps=30, fps=30):
    h, w = shape
    num_pixels = h * w

    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    coords = np.stack((ys, xs), axis=-1).reshape((num_pixels, 2))

    start_coords = coords
    end_coords = coords[np.argsort(mapping.reshape(-1))]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output, fourcc, fps, (w, h))

    for step in range(steps):
        alpha = step / (steps - 1)
        interp_coords = (1 - alpha) * start_coords + alpha * end_coords
        interp_coords_rounded = np.round(interp_coords).astype(int)
        interp_coords_rounded = np.clip(interp_coords_rounded, 0, [h - 1, w - 1])

        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[interp_coords_rounded[:, 0], interp_coords_rounded[:, 1]] = source_pixels

        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video.release()


MAX_SIZE = 512  # max dimension to resize to

source_img = load_and_resize_image("1.jpg", MAX_SIZE)
target_img = load_and_resize_image("2.jpg", MAX_SIZE)

mapping, source_pixels = fast_pixel_mapping(source_img, target_img)

create_transition_video(mapping, source_pixels, source_img.shape[:2], output="transition.mp4", steps=300, fps=30)
