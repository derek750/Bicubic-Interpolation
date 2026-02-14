import cv2
import numpy as np
import sys
import os


def cubic_weights(t):
    tt = t * t
    ttt = tt * t
    return np.array([
        (-ttt + 2 * tt - t) / 2,
        (3 * ttt - 5 * tt + 2) / 2,
        (-3 * ttt + 4 * tt + t) / 2,
        (ttt - tt) / 2
    ])

try:
    if len(sys.argv) != 3:
        raise Exception("Usage: python3 bicubic.py path/to/image.png scale_factor")

    input_file = sys.argv[1]
    scale = float(sys.argv[2])

    if scale <= 0:
        raise Exception("Scale factor must be > 0")

    if not os.path.isfile(input_file):
        raise Exception(f"Input file '{input_file}' does not exist")

    image = cv2.imread(input_file)
    if image is None:
        raise Exception("Failed to load image")

    height, width, _ = image.shape
    b, g, r = cv2.split(image)

    new_h = int(height * scale)
    new_w = int(width * scale)

    r_up = np.zeros((new_h, new_w), dtype=np.float32)
    g_up = np.zeros((new_h, new_w), dtype=np.float32)
    b_up = np.zeros((new_h, new_w), dtype=np.float32)

    for i in range(new_h):
        for j in range(new_w):
            src_y = i / scale
            src_x = j / scale

            y = int(np.floor(src_y))
            x = int(np.floor(src_x))

            # Boundary handling
            if y < 1 or y >= height - 2 or x < 1 or x >= width - 2:
                yy = min(max(y, 0), height - 1)
                xx = min(max(x, 0), width - 1)
                r_up[i, j] = r[yy, xx]
                g_up[i, j] = g[yy, xx]
                b_up[i, j] = b[yy, xx]
                continue

            wx = cubic_weights(src_x - x)
            wy = cubic_weights(src_y - y)

            r_patch = r[y-1:y+3, x-1:x+3]
            g_patch = g[y-1:y+3, x-1:x+3]
            b_patch = b[y-1:y+3, x-1:x+3]

            r_up[i, j] = np.sum(r_patch * wy[:, None] * wx[None, :])
            g_up[i, j] = np.sum(g_patch * wy[:, None] * wx[None, :])
            b_up[i, j] = np.sum(b_patch * wy[:, None] * wx[None, :])

    r_up = np.clip(r_up, 0, 255).astype(np.uint8)
    g_up = np.clip(g_up, 0, 255).astype(np.uint8)
    b_up = np.clip(b_up, 0, 255).astype(np.uint8)

    output = cv2.merge([b_up, g_up, r_up])

    os.makedirs("output", exist_ok=True)
    output_file = os.path.join(
        "output",
        os.path.splitext(os.path.basename(input_file))[0] + f"_x{scale}.jpg"
    )

    cv2.imwrite(output_file, output)
    print(f"Saved upscaled image to {output_file}")

except Exception as e:
    print("Error:", e)
