"""Step 4: audit the extracted landmarks and report how missing and abnormal rows are handled.

Training uses only rows with ``detected = 1``. Test and Unseen keep their undetected rows so that
evaluate.py counts them as wrong answers in the end-to-end accuracy. train.py and evaluate.py apply
these rules through ``features.normalize_df``; this script measures the data and writes the tables,
it does not rewrite the CSVs.

Outputs:
    results/detection_by_class.csv   images, rows and detection rate per class and split
    results/cleaning_summary.csv     row counts and anomaly checks per split
    results/undetected_by_class.png  bar chart of undetected images per class and split
    reports/figures/undetected_images.png
        contact sheet of every undetected image; it shows dataset images, so it is written to the
        git-ignored reports folder
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import CLASSES, SPLITS, get_image_paths, get_raw_base
from features import LM_COLS, normalize
from plotting import class_labels, plt

FAR_LOW, FAR_HIGH = -0.5, 1.5
X_COLS = [c for c in LM_COLS if c.startswith("x")]
Y_COLS = [c for c in LM_COLS if c.startswith("y")]
RESULTS = ROOT / "results"
FIGURES = ROOT / "reports" / "figures"
FONT_PATH = "/System/Library/Fonts/Supplemental/Tahoma.ttf"


def load_frames() -> dict:
    """Return ``{split_key: DataFrame}`` read from data/processed/landmarks_<split>.csv."""
    return {s: pd.read_csv(ROOT / "data" / "processed" / f"landmarks_{s}.csv") for s in SPLITS}


def count_images(raw_base: Path) -> dict:
    """Return ``{(split_key, code): number of JPEG files on disk}``."""
    return {(s, code): len(get_image_paths(raw_base, s, folder)) for s in SPLITS for code, _, folder in CLASSES}


def detection_by_class(frames: dict, image_counts: dict) -> pd.DataFrame:
    """Count images, CSV rows and detections for every class in every split.

    Args:
        frames: ``{split_key: landmarks DataFrame}``.
        image_counts: ``{(split_key, code): JPEG files on disk}``; images missing from the CSV are
            the ones extract_landmarks.py could not decode.

    Returns:
        One row per split and class with ``detection_rate = detected / rows``.
    """
    records = []
    for split, df in frames.items():
        for code, letter, _ in CLASSES:
            rows = df[df["label"] == code]
            detected = int(rows["detected"].sum())
            records.append({
                "split": split, "code": code, "letter": letter,
                "images": image_counts[(split, code)], "rows": len(rows),
                "unreadable": image_counts[(split, code)] - len(rows),
                "detected": detected, "undetected": len(rows) - detected,
                "detection_rate": round(detected / len(rows), 4) if len(rows) else np.nan,
            })
    return pd.DataFrame(records)


def anomaly_summary(df: pd.DataFrame) -> dict:
    """Run the step-4 checks on the detected rows of one split.

    Returns:
        Counts of rows with NaN coordinates, x/y far outside the image (below FAR_LOW or above
        FAR_HIGH), x/y slightly outside 0-1 only, rows whose 63 coordinates exactly repeat another
        row, and degenerate rows that ``normalize`` rejects; plus the file names of the duplicates.
    """
    det = df[df["detected"] == 1]
    xy = det[X_COLS + Y_COLS]
    far = ((xy < FAR_LOW) | (xy > FAR_HIGH)).any(axis=1)
    outside = ((xy < 0) | (xy > 1)).any(axis=1)
    duplicated = det.duplicated(subset=LM_COLS, keep=False)
    complete = det[~det[LM_COLS].isna().any(axis=1)]
    degenerate = sum(normalize(r, int(r["img_w"]), int(r["img_h"])) is None for _, r in complete.iterrows())
    return {
        "nan_rows": int(det[LM_COLS].isna().any(axis=1).sum()),
        "far_outside": int(far.sum()),
        "slightly_outside": int((outside & ~far).sum()),
        "duplicate_rows": int(duplicated.sum()),
        "degenerate": int(degenerate),
        "duplicate_files": sorted(Path(p).name for p in det.loc[duplicated, "file_path"]),
    }


def cleaning_summary(frames: dict, table: pd.DataFrame) -> pd.DataFrame:
    """Combine row counts, anomaly checks and class balance into one row per split.

    ``rows_used`` is what the later steps actually use: detected, non-degenerate rows for Training,
    and every row for Test and Unseen, where undetected images count as wrong predictions.
    """
    records = []
    for split, df in frames.items():
        part = table[table["split"] == split]
        checks = anomaly_summary(df)
        detected = int(part["detected"].sum())
        records.append({
            "split": split,
            "images": int(part["images"].sum()), "rows": int(part["rows"].sum()),
            "unreadable": int(part["unreadable"].sum()),
            "detected": detected, "undetected": int(part["undetected"].sum()),
            **{k: v for k, v in checks.items() if k != "duplicate_files"},
            "rows_used": detected - checks["degenerate"] - checks["nan_rows"] if split == "train" else len(df),
            "class_min": int(part["detected"].min()), "class_max": int(part["detected"].max()),
            "duplicate_files": " ".join(checks["duplicate_files"]),
        })
    return pd.DataFrame(records)


def plot_undetected(table: pd.DataFrame, path: Path):
    """Save a grouped bar chart of undetected images per class, one bar per split."""
    codes = [code for code, _, _ in CLASSES]
    splits = list(SPLITS)
    x = np.arange(len(codes))
    width = 0.8 / len(splits)
    fig, ax = plt.subplots(figsize=(11, 4))
    for i, split in enumerate(splits):
        counts = table[table["split"] == split].set_index("code").loc[codes, "undetected"]
        ax.bar(x + (i - (len(splits) - 1) / 2) * width, counts, width, label=split)
    ax.set_xticks(x, class_labels(codes))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("undetected images")
    ax.set_title("Images where MediaPipe found no hand")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def contact_sheet(frames: dict, path: Path, thumb_h: int = 200, cols: int = 4):
    """Save one image showing every undetected photo with its split, class and file name."""
    rows = pd.concat([df[df["detected"] == 0] for df in frames.values()])
    try:
        font = ImageFont.truetype(FONT_PATH, 18)
    except OSError:
        font = ImageFont.load_default()
    thumbs = []
    for _, r in rows.iterrows():
        img = Image.open(r["file_path"]).convert("RGB")
        img = img.resize((int(img.width * thumb_h / img.height), thumb_h))
        ImageDraw.Draw(img).text((6, 4), f"{r['split']} {r['label']} {Path(r['file_path']).name}",
                                 font=font, fill=(255, 255, 0))
        thumbs.append(img)
    cell_w = max(t.width for t in thumbs)
    sheet = Image.new("RGB", (cell_w * cols, thumb_h * -(-len(thumbs) // cols)), (30, 30, 30))
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % cols) * cell_w, (i // cols) * thumb_h))
    sheet.save(path)


def main():
    raw = get_raw_base(ROOT)
    if not raw.exists():
        sys.exit(f"ไม่พบ dataset ที่ {raw}")
    frames = load_frames()
    table = detection_by_class(frames, count_images(raw))
    summary = cleaning_summary(frames, table)

    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / "detection_by_class.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(RESULTS / "cleaning_summary.csv", index=False, encoding="utf-8-sig")
    plot_undetected(table, RESULTS / "undetected_by_class.png")
    contact_sheet(frames, FIGURES / "undetected_images.png")

    print(summary.drop(columns="duplicate_files").to_string(index=False))
    print("\nภาพที่หามือไม่เจอ ต่อคลาส:")
    print(table[table["undetected"] > 0][["split", "letter", "code", "undetected", "rows"]].to_string(index=False))
    print("\nบันทึก results/detection_by_class.csv, results/cleaning_summary.csv, results/undetected_by_class.png")
    print("บันทึก reports/figures/undetected_images.png (ภาพจาก dataset ไม่ขึ้น git)")


if __name__ == "__main__":
    main()
