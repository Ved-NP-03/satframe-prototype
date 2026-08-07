import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import json
import os

# ──────────────────────────────────────────
# SETTINGS — don't change anything else
# ──────────────────────────────────────────
DATA_FOLDER      = "data/goes19"
COMPARISON_FOLDER = "images/comparison"
ANIMATION_FOLDER  = "images/animation"
METRICS_FILE      = "metrics.json"

FRAME_FILES = [
    "frame_01.nc",
    "frame_02.nc",
    "frame_03.nc",
    "frame_04.nc",
    "frame_05.nc",
    "frame_06.nc",
]

TIMESTAMPS = [
    "2026-01-01  05:34 UTC",
    "2026-01-01  05:39 UTC",
    "2026-01-01  05:44 UTC",
    "2026-01-01  05:49 UTC",
    "2026-01-01  05:54 UTC",
    "2026-01-01  05:59 UTC",
]

# ──────────────────────────────────────────
# STEP 1 — Load one .nc file and extract data
# ──────────────────────────────────────────
def load_frame(filepath):
    """
    Opens a GOES-19 .nc file and extracts the
    brightness temperature (or radiance) as a 2D numpy array.
    Normalizes values to 0-1 range.
    """
    print(f"  Loading: {filepath}")
    ds = xr.open_dataset(filepath)

    # GOES-19 ABI stores radiance in variable called 'Rad'
    # We convert it to a numpy array
    if 'Rad' in ds:
        data = ds['Rad'].values
    else:
        # fallback — take first 2D variable found
        for var in ds.data_vars:
            arr = ds[var].values
            if arr.ndim == 2:
                data = arr
                break

    ds.close()

    # Replace fill values / bad pixels with NaN then 0
    data = np.where(np.isfinite(data), data, 0).astype(np.float32)

    # Normalize to 0-1
    dmin = data.min()
    dmax = data.max()
    if dmax - dmin > 0:
        data = (data - dmin) / (dmax - dmin)
    else:
        data = np.zeros_like(data)

    return data


# ──────────────────────────────────────────
# STEP 2 — Save a frame as a PNG image
# ──────────────────────────────────────────
def save_image(array, filepath, title="", colormap="gray"):
    """
    Saves a 2D numpy array as a PNG image.
    Uses a thermal colormap for satellite look.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(array, cmap=colormap, vmin=0, vmax=1)
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=10, pad=6,
                     color='white', fontweight='bold')
    fig.patch.set_facecolor('#0A0E1A')
    plt.tight_layout(pad=0.3)
    plt.savefig(filepath, dpi=100, bbox_inches='tight',
                facecolor='#0A0E1A')
    plt.close()
    print(f"  Saved: {filepath}")


# ──────────────────────────────────────────
# STEP 3 — Linear interpolation (simple average)
# ──────────────────────────────────────────
def linear_interpolate(frame_a, frame_b):
    """
    The simplest possible interpolation:
    just average the two frames pixel by pixel.
    This is our BASELINE — RIFE will beat this.
    """
    return (frame_a.astype(np.float32) + frame_b.astype(np.float32)) / 2.0


# ──────────────────────────────────────────
# STEP 4 — Calculate quality metrics
# ──────────────────────────────────────────
def calculate_metrics(predicted, ground_truth):
    """
    Compares predicted frame with real frame.
    Returns MSE, PSNR, SSIM scores.
    """
    pred = predicted.astype(np.float32)
    real = ground_truth.astype(np.float32)

    # MSE — mean squared error (lower is better)
    mse_val = float(np.mean((pred - real) ** 2))

    # PSNR — peak signal to noise ratio (higher is better)
    if mse_val > 0:
        psnr_val = float(psnr(real, pred, data_range=1.0))
    else:
        psnr_val = 100.0

    # SSIM — structural similarity (closer to 1 is better)
    ssim_val = float(ssim(real, pred, data_range=1.0))

    return {
        "MSE":  round(mse_val, 6),
        "PSNR": round(psnr_val, 2),
        "SSIM": round(ssim_val, 4)
    }


# ──────────────────────────────────────────
# MAIN — runs everything
# ──────────────────────────────────────────
def main():
    print("\n🛰️  SatFrame-AI Prototype — Processing Script")
    print("=" * 50)

    # Make sure output folders exist
    os.makedirs(COMPARISON_FOLDER, exist_ok=True)
    os.makedirs(ANIMATION_FOLDER, exist_ok=True)

    # ── Load all 6 frames ──
    print("\n[1/4] Loading satellite frames...")
    frames = []
    for fname in FRAME_FILES:
        path = os.path.join(DATA_FOLDER, fname)
        frame = load_frame(path)
        frames.append(frame)
    print(f"  ✅ Loaded {len(frames)} frames")
    print(f"  Frame shape: {frames[0].shape}")

    # ── COMPARISON SECTION ──
    # Use frames 1, 2, 3 (indices 0, 1, 2)
    # Input:  frame_01 (T=0) + frame_03 (T=10 min)
    # Predict: frame_02 (T=5 min) ← ground truth
    # Linear average as prediction
    print("\n[2/4] Generating comparison images...")

    frame_t0   = frames[0]   # 05:34
    frame_real = frames[1]   # 05:39 — GROUND TRUTH (what we want to predict)
    frame_t2   = frames[2]   # 05:44

    # Linear interpolation prediction
    frame_pred = linear_interpolate(frame_t0, frame_t2)

    # Save comparison images
    save_image(frame_t0,   f"{COMPARISON_FOLDER}/frame_t0.png",
               title=f"Input Frame A\n{TIMESTAMPS[0]}", colormap="inferno")

    save_image(frame_pred, f"{COMPARISON_FOLDER}/frame_predicted.png",
               title=f"Predicted Middle Frame\n(Linear Avg — Baseline)",
               colormap="inferno")

    save_image(frame_real, f"{COMPARISON_FOLDER}/frame_real.png",
               title=f"Real Middle Frame\n{TIMESTAMPS[1]}", colormap="inferno")

    save_image(frame_t2,   f"{COMPARISON_FOLDER}/frame_t2.png",
               title=f"Input Frame B\n{TIMESTAMPS[2]}", colormap="inferno")

    print("  ✅ Comparison images saved")

    # ── METRICS ──
    print("\n[3/4] Calculating quality metrics...")
    metrics = calculate_metrics(frame_pred, frame_real)
    print(f"  MSE  = {metrics['MSE']}  (lower is better)")
    print(f"  PSNR = {metrics['PSNR']} dB  (higher is better, target >30)")
    print(f"  SSIM = {metrics['SSIM']}  (closer to 1 is better)")

    # Save metrics to JSON for the webpage to load
    metrics_output = {
        "method": "Linear Interpolation (Baseline)",
        "note": "RIFE deep learning model will significantly improve these scores",
        "scores": metrics,
        "timestamps": {
            "input_A":    TIMESTAMPS[0],
            "predicted":  TIMESTAMPS[1] + " (predicted)",
            "ground_truth": TIMESTAMPS[1] + " (real)",
            "input_B":    TIMESTAMPS[2]
        }
    }
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics_output, f, indent=2)
    print(f"  ✅ Metrics saved to {METRICS_FILE}")

    # ── ANIMATION SECTION ──
    # Save all 6 original frames for animation
    # Also save 5 interpolated frames between each pair
    print("\n[4/4] Generating animation frames...")

    # Original frames (6 frames, every 5 min)
    for i, (frame, ts) in enumerate(zip(frames, TIMESTAMPS)):
        save_image(frame,
                   f"{ANIMATION_FOLDER}/orig_{i+1:02d}.png",
                   title=ts, colormap="inferno")

    # Interpolated frames (11 frames — original 6 + 5 in-between)
    # Frame sequence: orig1, interp1-2, orig2, interp2-3, orig3...
    interp_idx = 1
    for i in range(len(frames)):
        # Save the real frame
        save_image(frames[i],
                   f"{ANIMATION_FOLDER}/interp_{interp_idx:02d}.png",
                   title=TIMESTAMPS[i], colormap="inferno")
        interp_idx += 1

        # If not the last frame, save an interpolated one after it
        if i < len(frames) - 1:
            between = linear_interpolate(frames[i], frames[i+1])
            # Calculate approximate midpoint timestamp label
            save_image(between,
                       f"{ANIMATION_FOLDER}/interp_{interp_idx:02d}.png",
                       title="AI Interpolated Frame", colormap="inferno")
            interp_idx += 1

    print(f"  ✅ Animation frames saved")

    # ── DONE ──
    print("\n" + "=" * 50)
    print("✅ ALL DONE! Open index.html in your browser.")
    print(f"\nSummary:")
    print(f"  Comparison images → {COMPARISON_FOLDER}/")
    print(f"  Animation frames  → {ANIMATION_FOLDER}/")
    print(f"  Metrics           → {METRICS_FILE}")
    print(f"\nMetric Scores (Linear Baseline):")
    print(f"  MSE  = {metrics['MSE']}")
    print(f"  PSNR = {metrics['PSNR']} dB")
    print(f"  SSIM = {metrics['SSIM']}")
    print("\n🚀 During hackathon, RIFE will beat all these scores!")


if __name__ == "__main__":
    main()