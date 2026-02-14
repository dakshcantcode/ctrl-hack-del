"""
Speed-heatmap generation from drawing strokes.
"""
import numpy as np
import cv2
from utils.image_processing import image_to_base64


# ── Speed heatmap ─────────────────────────────────────────────

def generate_speed_heatmap(strokes: list[dict], canvas_size: tuple) -> str:
    """
    Generate a heatmap where colour = local speed variability.
    Green = steady (healthy), Red = erratic (tremor indicator).
    Returns base64-encoded PNG data URL.
    """
    if len(strokes) < 3:
        blank = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
        return image_to_base64(blank)

    width, height = canvas_size
    heatmap = np.zeros((height, width, 3), dtype=np.uint8)

    pts = np.array([[s["x"], s["y"]] for s in strokes])
    ts = np.array([s["t"] for s in strokes])

    dists = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    dt = np.diff(ts) / 1000.0
    dt[dt == 0] = 1e-6
    speeds = dists / dt

    for i in range(len(speeds)):
        lo = max(0, i - 5)
        hi = min(len(speeds), i + 5)
        local_speeds = speeds[lo:hi]
        local_cv = float(np.std(local_speeds) / (np.mean(local_speeds) + 1e-6))
        intensity = min(local_cv / 1.5, 1.0)

        colour = (
            int(intensity * 255),           # Red
            int((1 - intensity) * 255),     # Green
            0,                               # Blue
        )

        pt1 = (int(pts[i][0]), int(pts[i][1]))
        pt2 = (int(pts[i + 1][0]), int(pts[i + 1][1]))
        cv2.line(heatmap, pt1, pt2, colour, 3)

    return image_to_base64(heatmap)
