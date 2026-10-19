"""Step 2 EDA: check the class folders, count images per class and split, and look for images shared by Training and Test."""
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import CLASSES, SPLITS, get_image_paths, get_raw_base, verify_classes

RAW = get_raw_base(ROOT)


def file_hash(path: Path) -> str:
    """MD5 of the file bytes, used to spot identical images."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def count_images() -> dict:
    """Return ``{split_key: {thai_letter: image_count}}``."""
    counts: dict[str, dict[str, int]] = {}
    for split_key in SPLITS:
        counts[split_key] = {}
        for _, char, folder in CLASSES:
            paths = get_image_paths(RAW, split_key, folder)
            counts[split_key][char] = len(paths)
    return counts


def check_cross_split_duplicates() -> list[tuple[str, tuple[str, ...]]]:
    """Return ``(hash, splits)`` for every image that appears in more than one split."""
    splits_by_hash: dict[str, set[str]] = defaultdict(set)
    for split_key in SPLITS:
        for _, _, folder in CLASSES:
            for p in get_image_paths(RAW, split_key, folder):
                splits_by_hash[file_hash(p)].add(split_key)
    return [(h, tuple(sorted(s))) for h, s in splits_by_hash.items() if len(s) > 1]


def print_count_table(counts: dict):
    """Print the image counts as a class x split table with totals."""
    splits = list(SPLITS.keys())
    header = f"{'ตัว':>4}  " + "  ".join(f"{s:>8}" for s in splits)
    print(header)
    print("-" * len(header))
    totals = {s: 0 for s in splits}
    for _, char, _ in CLASSES:
        row = f"{char:>4}  " + "  ".join(f"{counts[s][char]:>8}" for s in splits)
        print(row)
        for s in splits:
            totals[s] += counts[s][char]
    print("-" * len(header))
    print(f"{'รวม':>4}  " + "  ".join(f"{totals[s]:>8}" for s in splits))


def main():
    print("=" * 50)
    print("ตรวจสอบคลาส")
    missing = verify_classes(RAW)
    if missing:
        print("⚠️  โฟลเดอร์หาย:", missing)
        sys.exit(1)
    print(f"✅  ครบ 15 คลาสทั้ง 3 split")

    print("\nจำนวนภาพ (ต่อคลาส × split)")
    counts = count_images()
    print_count_table(counts)

    totals = {s: sum(counts[s].values()) for s in SPLITS}
    print(f"\nคาดว่า Training 4,320 / Test 2,880 / Unseen 300")
    print(f"นับได้:  Training {totals['train']:,} / Test {totals['test']:,} / Unseen {totals['unseen']:,}")

    balanced = all(
        counts[s][char] == counts[s][list({c for _, c, _ in CLASSES})[0]]
        for s in ("train", "test")
        for _, char, _ in CLASSES
    )

    print("\nตรวจภาพซ้ำข้ามชุด (train / test / unseen) ...")
    dupes = check_cross_split_duplicates()
    if dupes:
        print(f"⚠️  พบ data leakage: {len(dupes)} ภาพ")
        for h, splits in dupes[:5]:
            print(f"   {h[:8]}  {' ↔ '.join(splits)}")
    else:
        print("✅  ไม่พบภาพซ้ำข้ามชุด")

    per_class_train = [counts["train"][char] for _, char, _ in CLASSES]
    per_class_test  = [counts["test"][char]  for _, char, _ in CLASSES]
    print(f"\nTrain ต่อคลาส: min={min(per_class_train)} max={max(per_class_train)}")
    print(f"Test  ต่อคลาส: min={min(per_class_test)}  max={max(per_class_test)}")


if __name__ == "__main__":
    main()
