# Thai Fingerspelling Recognition

Recognizes 15 Thai fingerspelling consonants — ก ด ต น บ พ ฟ ม ย ร ล ว ส ห อ — from a single image or
a live webcam. MediaPipe Hand Landmarker turns each hand into 21 3-D landmarks, a shared
normalization step makes them independent of position, distance and image aspect ratio, and a
scikit-learn classifier predicts the letter.

## Results

Final model: SVM (RBF kernel, `C=100`, `gamma=0.1`), selected on Test macro-F1 among the
experiments in `results/experiments.csv`.

| Split | Images | Hand detected | Accuracy (detected) | Macro-F1 | End-to-end accuracy |
|---|---:|---:|---:|---:|---:|
| Test | 1,440 | 1,431 (99.4%) | 0.9811 | 0.9810 | 0.9750 |
| Unseen | 300 | 299 (99.7%) | pending | pending | pending |

End-to-end accuracy counts images where no hand was detected as wrong answers. The Unseen split is
the final hold-out and is reported once the final evaluation is run (see
[Evaluation protocol](#evaluation-protocol)).

Test macro-F1 of every experiment (`results/experiments.csv`, chart in
`results/experiments_macro_f1.png`):

| Model | E1 raw | E2 normalized | E3 tuned | E4 mirror aug |
|---|---:|---:|---:|---:|
| KNN | 0.7854 | 0.9408 | 0.9756 | 0.9493 |
| SVM | 0.9565 | 0.9580 | **0.9810** | 0.9559 |
| Random Forest | 0.9369 | 0.9595 | 0.9588 | 0.9602 |
| MLP | 0.9657 | 0.9762 | 0.9769 | 0.9706 |

E1 uses the coordinates exactly as MediaPipe returns them; E2–E4 use `features.normalize`, and
only those are eligible for selection. Per-class scores and the confusion matrix of the final model
are in `results/eval_test.md` and `results/eval_test_confusion.png`.

## Repository layout

```
src/
  config.py             dataset layout, class table, random seed
  extract_landmarks.py  step 3 — MediaPipe landmarks for every image -> data/processed/*.csv
  clean.py              step 4 — data audit: detection per class, anomaly checks, charts
  features.py           step 5 — the one normalization used by training, evaluation and demos
  train.py              step 6 — experiments E1–E4 and model selection on Test
  evaluate.py           step 7 — per-split metrics, per-class tables, confusion matrices, charts
  inference.py          landmarks -> class probabilities, shared by both demos
  realtime.py           step 8 — live webcam demo with a measured FPS counter
  demo_quiz.py          quiz mode with reference images, a smoke test for the live pipeline
  plotting.py           matplotlib setup with a Thai-capable font
  fonts.py              Thai font lookup for Pillow on macOS, Windows and Linux
tests/                  pytest suite (also runs in Docker)
notebooks/01_eda.py     step 2 — image counts and cross-split duplicate check
data/raw/               One-Stage-TFS images and annotations (tracked, about 690 MB)
data/processed/         extracted landmarks (tracked)
models/                 best_meta.json (tracked); trained models and hand_landmarker.task (ignored)
results/                experiment log, data-audit tables and charts
reports/                course reports, kept locally (ignored)
```

## Getting started

Tested with Python 3.9 on macOS (Apple Silicon). The Docker image gives teammates on Linux or
Windows the same environment; the test suite is verified inside it.

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**MediaPipe model** — download once into `models/`:

```bash
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

On Windows PowerShell use `Invoke-WebRequest -Uri <url> -OutFile models/hand_landmarker.task`.

**Dataset** — One-Stage-TFS from Mendeley Data
([10.17632/rknd3wbz42.1](https://doi.org/10.17632/rknd3wbz42.1)) is included under
`data/raw/One-Stage-TFS Thai One-Stage Fingerspelling Dataset/` (`Training set`, `Test set`,
`Unseen set`), so a clone is about 690 MB. Keep this repository private unless the dataset license
allows redistribution.

## Usage

The extracted landmarks are committed too, so extraction only needs to run again after changing
`extract_landmarks.py` or the dataset.

| Step | Command | Writes |
|---|---|---|
| Extract landmarks | `python src/extract_landmarks.py` | `data/processed/landmarks_{train,test,unseen}.csv` |
| Audit the data | `python src/clean.py` | `results/detection_by_class.csv`, `cleaning_summary.csv`, `undetected_by_class.png` |
| Train and select | `python src/train.py` | `models/best.joblib`, `models/best_meta.json`, `results/experiments.csv` |
| Evaluate on Test | `python src/evaluate.py --splits test` | `results/eval_test_*`, `results/experiments_macro_f1.png` |
| Final evaluation | `python src/evaluate.py --splits unseen` | `results/eval_unseen_*` — run once, after choosing the model |
| Live demo | `python src/realtime.py` | — (press `q` to quit; prints the average FPS) |
| Quiz mode | `python src/demo_quiz.py` | — (`Space` skips a letter, `q` quits) |

The webcam demos must run on the host. On macOS, grant the terminal app camera access
(System Settings → Privacy & Security → Camera) and restart it.

### Docker

```bash
docker compose run --rm test     # pytest in Linux
docker compose run --rm train    # extract -> train -> evaluate (Test and Unseen)
```

Building the image needs `models/hand_landmarker.task`. Docker cannot reach the webcam on macOS
or Windows, so the demos stay on the host.

## Tests

```bash
python -m pytest tests/
```

| File | Guards |
|---|---|
| `test_features.py` | translation, scale and aspect-ratio invariance, degenerate hands, mirroring, row filtering in `normalize_df` |
| `test_inference.py` | the demo inference path, replayed on the labelled Test split, matches `predict` |
| `test_train.py` | raw/normalized row alignment for E1, mirror augmentation, the MLP configuration |
| `test_evaluate.py` | per-class tables and confusion matrices are attached to the right classes |
| `test_clean.py` | the step-4 anomaly checks and per-class counts |
| `test_config.py` | image discovery for both folder layouts and both `.jpg` / `.JPG` extensions |
| `test_fonts.py` | Thai font lookup and its fallback when no font is installed |
| `test_realtime.py` | the FPS meter, with a fake clock |

The normalization, inference and training tests were checked with 11 deliberately injected faults
(mutation testing); each fault makes at least one test fail.

## Design notes

- **One normalization.** `features.normalize` corrects the aspect ratio (`x, z × W/H`), moves the
  wrist to the origin and divides by the wrist–middle-finger-MCP distance. Training, evaluation and
  both demos call the same function.
- **Class order.** `predict_proba` columns follow `clf.classes_` (sorted codes), not the Thai
  order of `config.CLASSES`. Both demos predict through `inference.py`, which keys probabilities
  by class name, and `test_inference.py` checks that this path matches `predict` on the Test split.
- **Mirroring.** The live demo mirrors frames before detection. The dataset contains both hand
  orientations (about 70/30 by MediaPipe handedness) and the model scores 98.5% and 97.5% on the
  two orientations of the Test split, so either hand works.
- **Missing and abnormal data.** Every class has 384 Training, 96 Test and 20 Unseen images (some
  saved as `.JPG`, which image discovery matches case-insensitively). Training uses detected rows
  only; Test and Unseen keep undetected images so they count against end-to-end accuracy. One
  unreadable Training image (`sor55.jpg`) is skipped at extraction. There are no NaN, far-out-of-frame or degenerate rows. One duplicate image
  pair exists in Training and one in Test; removing the Training duplicate changes no Test
  prediction, so the data is kept as extracted. Six of the 16 undetected images are the
  downward-pointing sign พ.

## Evaluation protocol

| Split | Used for |
|---|---|
| Training | fitting and cross-validated grid search |
| Test | comparing experiments and choosing the final model |
| Unseen | one final measurement of the chosen model |

## Known limitations

- `MLPClassifier` runs without `early_stopping`: scikit-learn 1.6.1 fails when it is combined
  with string labels.
- Augmentation (E4) covers mirroring only; rotation and Gaussian noise are not implemented yet.
- The live frame rate is bounded by the camera: processing takes about 10 ms per 1280×720 frame
  (median, Apple M4 Pro), so check the course target of 15 FPS with the average that `realtime.py`
  prints on exit, on the machine used for the presentation.
- The course recommends Python 3.11; the code is tested on 3.9.

## References

- Lata & Surinta (2022). *ICIC Express Letters*. DOI
  [10.24507/icicel.16.05.529](https://doi.org/10.24507/icicel.16.05.529)
- One-Stage-TFS dataset. DOI [10.17632/rknd3wbz42.1](https://doi.org/10.17632/rknd3wbz42.1)
- MediaPipe Hand Landmarker (Tasks API)
