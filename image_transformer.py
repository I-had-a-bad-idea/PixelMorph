import numpy as np
import cv2
import time
import sys

def load_and_resize_image(path, size):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_y = (size - new_h) // 2
    pad_x = (size - new_w) // 2
    padded = np.zeros((size, size, 3), dtype=np.uint8)
    padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = img

    return padded

def fast_pixel_mapping(source_img, target_img):
    # Flatten and compute brightness for sorting
    source_flat = source_img.reshape(-1, 3)
    target_flat = target_img.reshape(-1, 3)

    # Use luminance instead of RGB sum for slightly better visual results
    source_lum = source_flat @ np.array([0.2126, 0.7152, 0.0722])
    target_lum = target_flat @ np.array([0.2126, 0.7152, 0.0722])

    source_order = np.argsort(source_lum)
    target_order = np.argsort(target_lum)

    mapping = np.empty_like(target_order)
    mapping[target_order] = source_order
    return mapping.reshape(source_img.shape[:2]), source_flat

def create_transition_video(mapping, source_pixels, shape, output="transition.mp4",
                            steps=30, fps=30, hold_duration_sec=2):
    h, w = shape
    num_pixels = h * w
    hold_frames = int(hold_duration_sec * fps)

    ys, xs = np.mgrid[0:h, 0:w]
    coords = np.column_stack((ys.ravel(), xs.ravel()))
    end_coords = coords[np.argsort(mapping.ravel())]

    video = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    for step in range(steps):
        alpha = step / (steps - 1)
        interp = np.round((1 - alpha) * coords + alpha * end_coords).astype(int)
        interp[:, 0] = np.clip(interp[:, 0], 0, h - 1)
        interp[:, 1] = np.clip(interp[:, 1], 0, w - 1)

        frame.fill(0)
        frame[interp[:, 0], interp[:, 1]] = source_pixels
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(bgr)

        if step == 0 or step == steps - 1:
            for _ in range(hold_frames):
                video.write(bgr)

    video.release()

def main():
    img_1_path, img_2_path, output_video_path = sys.argv[1:4]
    MAX_SIZE = 512

    t0 = time.time()
    print("Loading images...")
    source_img = load_and_resize_image(img_1_path, MAX_SIZE)
    target_img = load_and_resize_image(img_2_path, MAX_SIZE)

    print("Computing pixel mapping...")
    t1 = time.time()
    mapping, source_pixels = fast_pixel_mapping(source_img, target_img)
    print(f"Mapping done in {time.time() - t1:.2f}s")

    print("Generating video...")
    create_transition_video(mapping, source_pixels, source_img.shape[:2],
                            output=output_video_path, steps=150, fps=30, hold_duration_sec=1)
    print(f"Video saved to {output_video_path}")
    print(f"Total time: {time.time() - t0:.2f}s")

if __name__ == "__main__":
    main()
