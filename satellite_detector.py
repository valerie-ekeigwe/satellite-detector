import argparse
import csv
from collections import deque
from dataclasses import dataclass, field
from math import hypot

import cv2
import numpy as np

BG_WINDOW = 31          # frames in the background buffer
THRESH_Z = 3.5          # 3.0 was picking up sensor noise on darker clips
MIN_AREA, MAX_AREA = 2, 30
LINK_DIST = 12          # px a blob can move between frames and still match
MAX_MISSES = 2
MIN_TRACK = 8
SPEED_RANGE = (5, 500)  # px/s, planes and noise tend to land outside this
MAX_SPEED_CV = 0.4      # satellites move at a near constant speed


def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cv2.blur(gray, (3, 3))


class Background:
    # median over a sliding window: stars sit still so they survive,
    # a satellite is only in each pixel for a frame or two so it gets voted out
    def __init__(self, size):
        self.frames = deque(maxlen=size)

    def update(self, gray):
        self.frames.append(gray)

    def ready(self):
        return len(self.frames) >= self.frames.maxlen // 2

    def get(self):
        return np.median(np.stack(self.frames), axis=0).astype(np.uint8)


def motion_mask(diff, z=THRESH_Z):
    # robust z-score with MAD instead of std so a few bright pixels don't
    # drag the threshold up. 1.4826 scales MAD to match std for normal noise
    med = np.median(diff)
    mad = np.median(np.abs(diff - med)) + 1e-6
    mask = ((diff - med) / (1.4826 * mad) > z).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(cv2.erode(mask, kernel), kernel)


def find_blobs(mask):
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [tuple(centroids[i]) for i in range(1, n)
            if MIN_AREA <= stats[i, cv2.CC_STAT_AREA] <= MAX_AREA]


@dataclass
class Track:
    id: int
    points: list = field(default_factory=list)
    misses: int = 0

    def predict(self):
        if len(self.points) < 2:
            return self.points[-1][1:]
        (_, x1, y1), (_, x2, y2) = self.points[-2:]
        return 2 * x2 - x1, 2 * y2 - y1

    def is_satellite(self, fps):
        if len(self.points) < MIN_TRACK:
            return False
        speeds = np.array([hypot(b[1] - a[1], b[2] - a[2])
                           for a, b in zip(self.points, self.points[1:])]) * fps
        mean = speeds.mean()
        return SPEED_RANGE[0] < mean < SPEED_RANGE[1] and speeds.std() / mean < MAX_SPEED_CV


class Tracker:
    def __init__(self):
        self.tracks = []
        self.finished = []
        self.next_id = 1

    def update(self, frame_no, blobs):
        unmatched = set(range(len(blobs)))
        for t in self.tracks:
            px, py = t.predict()
            best = min(unmatched, key=lambda j: hypot(blobs[j][0] - px, blobs[j][1] - py), default=None)
            if best is not None and hypot(blobs[best][0] - px, blobs[best][1] - py) <= LINK_DIST:
                t.points.append((frame_no, *blobs[best]))
                t.misses = 0
                unmatched.remove(best)
            else:
                t.misses += 1

        for j in unmatched:
            self.tracks.append(Track(self.next_id, [(frame_no, *blobs[j])]))
            self.next_id += 1
        self.finished += [t for t in self.tracks if t.misses > MAX_MISSES]
        self.tracks = [t for t in self.tracks if t.misses <= MAX_MISSES]


def draw(frame, tracker, confirmed, frame_no):
    for t in tracker.tracks:
        colour = (0, 220, 0) if t.id in confirmed else (160, 160, 160)
        pts = np.array([(int(x), int(y)) for _, x, y in t.points[-30:]])
        cv2.polylines(frame, [pts], False, colour, 1)
        x, y = pts[-1]
        cv2.putText(frame, str(t.id), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)
    cv2.putText(frame, f"frame {frame_no}  satellites {len(confirmed)}", (8, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return frame


def main():
    ap = argparse.ArgumentParser(description="find satellites in night sky video")
    ap.add_argument("video")
    ap.add_argument("--out", default="annotated.mp4")
    ap.add_argument("--csv", help="save tracks to a csv")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"couldn't open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    bg, tracker, confirmed = Background(BG_WINDOW), Tracker(), set()
    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = preprocess(frame)
        bg.update(gray)

        if bg.ready():
            diff = cv2.absdiff(gray, bg.get())
            tracker.update(frame_no, find_blobs(motion_mask(diff)))
            for t in tracker.tracks:
                if t.id not in confirmed and t.is_satellite(fps):
                    confirmed.add(t.id)
                    print(f"satellite #{len(confirmed)} (track {t.id}) at frame {frame_no}")
            frame = draw(frame, tracker, confirmed, frame_no)

        out.write(frame)
        if args.show:
            cv2.imshow("satellite detector", frame)
            if cv2.waitKey(1) == 27:
                break
        frame_no += 1

    cap.release()
    out.release()
    if args.show:
        cv2.destroyAllWindows()
    print(f"done, {len(confirmed)} satellites")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["track", "frame", "x", "y"])
            for t in tracker.finished + tracker.tracks:
                if t.id in confirmed:
                    writer.writerows([t.id, *p] for p in t.points)


if __name__ == "__main__":
    main()
