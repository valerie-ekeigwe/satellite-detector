# satellite-detector

Finds satellites in night sky video. Started as a small class project and I kept going with it because it was fun watching it pick up Starlink passes.

## How it works

Stars barely move between frames, satellites do. So:

1. Keep a rolling median of the last 31 frames. Stars survive the median, a satellite only crosses each pixel for a frame or two so it drops out. That gives a clean "sky without satellites" background.
2. Subtract it from the current frame and threshold the difference with a robust z-score (median + MAD instead of mean + std, so a few hot pixels don't wreck the threshold).
3. Pull out small blobs and link them frame to frame, predicting where each one should be next from its last step.
4. A track counts as a satellite if it lasts long enough and moves at a steady speed. Planes blink and noise jitters, so both fail that check.

## Run it

```bash
pip install -r requirements.txt
python satellite_detector.py your_clip.mp4 --show --csv tracks.csv
```

You get `annotated.mp4` with the tracks drawn on (green = satellite) and, with `--csv`, every satellite track point by frame.

The tuning knobs are the constants at the top of the file.

## What I'd improve

- Satellites that fade in and out (tumbling ones) get split into two tracks. A Kalman filter would probably hold them together better than the straight-line guess I'm using now.
- The median background gets slow on 4K footage. Could update it incrementally instead of recomputing it every frame.
- It doesn't know which satellite it found. Matching tracks against TLE data from CelesTrak would let it label passes by name.
