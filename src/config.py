"""Dataset layout, class table and random seed shared by every script.

``CLASSES`` lists ``(code, Thai letter, dataset folder)`` in Thai alphabetical order. Trained
models order their outputs by sorted code instead (see ``inference.class_probabilities``).
"""
from pathlib import Path

DATASET_DIR = "One-Stage-TFS Thai One-Stage Fingerspelling Dataset"

SEED = 42

SPLITS = {
    "train":  "Training set",
    "test":   "Test set",
    "unseen": "Unseen set",
}

CLASSES = [
    ("ko_kai",    "ก", "KO_KAI"),
    ("do_dek",    "ด", "DOR_DEK"),
    ("to_tao",    "ต", "TOR_TAO"),
    ("no_nu",     "น", "NOR_NOO"),
    ("bo_baimai", "บ", "BOR_BAI_MAI"),
    ("po_phan",   "พ", "POR_PHAN"),
    ("fo_fan",    "ฟ", "FOR_FUN"),
    ("mo_ma",     "ม", "MOR_MAH"),
    ("yo_yak",    "ย", "YOR_YAK"),
    ("ro_ruea",   "ร", "RAO_RUEA"),
    ("lo_ling",   "ล", "LOR_LING"),
    ("wo_waen",   "ว", "WOR_WAEN"),
    ("so_suea",   "ส", "SOR_SUEA"),
    ("ho_hip",    "ห", "HOR_HEEP"),
    ("o_ang",     "อ", "OR_ANG"),
]

CLASS_ID   = {code: i for i, (code, _, _) in enumerate(CLASSES)}
FOLDER_MAP = {folder: code for code, _, folder in CLASSES}


def get_raw_base(project_root: Path) -> Path:
    """Return the folder under data/raw that the dataset archive was extracted to."""
    return project_root / "data" / "raw" / DATASET_DIR


def get_image_paths(raw_base: Path, split_key: str, folder: str) -> list[Path]:
    """Return the sorted JPEG paths of one class in one split.

    Training and Test keep images in an ``Images (JPEG)`` subfolder, while Unseen stores them
    directly in the class folder; both layouts are handled.
    """
    split_dir  = raw_base / SPLITS[split_key]
    class_dir  = split_dir / folder
    img_subdir = class_dir / "Images (JPEG)"
    if img_subdir.exists():
        return sorted(img_subdir.glob("*.jpg"))
    return sorted(class_dir.glob("*.jpg"))


def verify_classes(raw_base: Path) -> dict[str, list[str]]:
    """Return ``{split_key: [missing folders]}`` for class folders absent from a split; empty when complete."""
    missing: dict[str, list[str]] = {}
    for split_key, split_name in SPLITS.items():
        split_dir = raw_base / split_name
        found = {d.name for d in split_dir.iterdir() if d.is_dir()}
        expected = {folder for _, _, folder in CLASSES}
        diff = expected - found
        if diff:
            missing[split_key] = sorted(diff)
    return missing
