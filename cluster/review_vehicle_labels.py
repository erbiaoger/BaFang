import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.signal import butter, sosfiltfilt


def _add_repo_root_to_path() -> None:
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        if (parent / "pyproject.toml").exists():
            sys.path.insert(0, str(parent))
            return


_add_repo_root_to_path()

from cluster.agc_dataset import load_paired_dataset  # noqa: E402

try:  # noqa: E402
    from dasQt.process.filter import bandpass as _bandpass_impl
except Exception:  # pragma: no cover - optional local dependency
    def _bandpass_impl(signal, fs, freqmin, freqmax, corners=4, **_kwargs):
        nyquist = 0.5 * fs
        low = max(freqmin / nyquist, 1e-6)
        high = min(freqmax / nyquist, 0.999)
        sos = butter(corners, [low, high], btype="bandpass", output="sos")
        return sosfiltfilt(sos, signal)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review vehicle predictions with AGC/origin/bandpass plots")
    parser.add_argument("--dataset_npz", required=True, help="Dataset manifest from build_agc_training_set.py")
    parser.add_argument("--meta_csv", required=True, help="Metadata CSV from build_agc_training_set.py")
    parser.add_argument("--predictions_csv", required=True, help="Prediction CSV from predict_vehicle_classifier.py")
    parser.add_argument("--fs", type=float, default=1000.0)
    parser.add_argument("--freqmin", type=float, default=0.1)
    parser.add_argument("--freqmax", type=float, default=2.0)
    return parser.parse_args()


def _normalize(sig: np.ndarray) -> np.ndarray:
    sig = np.asarray(sig, dtype=float).reshape(-1)
    peak = np.max(np.abs(sig)) if sig.size else 0.0
    return sig / peak if peak != 0 else sig


def _read_predictions(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["sample_id"] = int(row["sample_id"])
            row["prob_small"] = float(row["prob_small"])
            row["rule_smallcar_high_conf"] = int(row.get("rule_smallcar_high_conf", 0) or 0)
            rows.append(row)
    return rows


class ReviewViewer(QMainWindow):
    def __init__(self, signals_origin, signals_agc, meta_rows, prediction_rows, fs=1000.0, freqmin=0.1, freqmax=2.0):
        super().__init__()
        self.setWindowTitle("Vehicle Prediction Review")
        self.setGeometry(100, 100, 1200, 920)

        self.signals_origin = signals_origin
        self.signals_agc = signals_agc
        self.meta_by_sample_id = {int(row["sample_id"]): row for row in meta_rows}
        self.samples = []
        for pred in prediction_rows:
            meta = self.meta_by_sample_id.get(int(pred["sample_id"]))
            if meta is None:
                continue
            merged = dict(meta)
            merged.update(pred)
            self.samples.append(merged)

        self.fs = fs
        self.freqmin = freqmin
        self.freqmax = freqmax
        self.time_mode = "absolute"
        self.current_index = 0
        self.filtered_samples = []

        self.label_combo = QComboBox()
        self.label_combo.addItem("All labels", "all")
        for label in ["small", "large", "uncertain"]:
            self.label_combo.addItem(label, label)

        self.confidence_combo = QComboBox()
        self.confidence_combo.addItem("All confidence", "all")
        for bucket in sorted({row["confidence_bucket"] for row in self.samples}):
            self.confidence_combo.addItem(bucket, bucket)

        self.cluster_combo = QComboBox()
        self.cluster_combo.addItem("All clusters", "all")
        cluster_ids = sorted({str(row.get("cluster_id_old", "")) for row in self.samples if str(row.get("cluster_id_old", "")) != ""})
        for cluster_id in cluster_ids:
            self.cluster_combo.addItem(f"Cluster {cluster_id}", cluster_id)

        self.sort_combo = QComboBox()
        self.sort_combo.addItem("Prob small desc", "prob_desc")
        self.sort_combo.addItem("Prob small asc", "prob_asc")
        self.sort_combo.addItem("Sample id asc", "sample_id")

        self.status_label = QLabel("")

        self.canvas = FigureCanvas(Figure(figsize=(8, 9)))
        self.ax_agc, self.ax_origin, self.ax_bandpass = self.canvas.figure.subplots(3, 1, sharex=True)

        controls = QHBoxLayout()
        controls.addWidget(self.label_combo)
        controls.addWidget(self.confidence_combo)
        controls.addWidget(self.cluster_combo)
        controls.addWidget(self.sort_combo)
        controls.addWidget(self.status_label)

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.addLayout(controls)
        layout.addWidget(self.canvas)
        self.setCentralWidget(central_widget)

        self.label_combo.currentIndexChanged.connect(self.on_filters_changed)
        self.confidence_combo.currentIndexChanged.connect(self.on_filters_changed)
        self.cluster_combo.currentIndexChanged.connect(self.on_filters_changed)
        self.sort_combo.currentIndexChanged.connect(self.on_filters_changed)
        self.canvas.mpl_connect("scroll_event", self.on_mouse_scroll)

        self.apply_filters()

    def _get_scroll_step(self, event):
        step = getattr(event, "step", 0)
        if step:
            return 1 if step > 0 else -1
        gui_event = getattr(event, "guiEvent", None)
        if gui_event is not None and hasattr(gui_event, "angleDelta"):
            delta = gui_event.angleDelta()
            raw = delta.y() if delta.y() != 0 else delta.x()
            if raw != 0:
                return 1 if raw > 0 else -1
        button = getattr(event, "button", None)
        if button in ("up", "right"):
            return 1
        if button in ("down", "left"):
            return -1
        return 0

    def apply_filters(self):
        label_value = self.label_combo.currentData()
        confidence_value = self.confidence_combo.currentData()
        cluster_value = self.cluster_combo.currentData()
        sort_value = self.sort_combo.currentData()

        filtered = self.samples
        if label_value != "all":
            filtered = [row for row in filtered if row["pred_label"] == label_value]
        if confidence_value != "all":
            filtered = [row for row in filtered if row["confidence_bucket"] == confidence_value]
        if cluster_value != "all":
            filtered = [row for row in filtered if str(row.get("cluster_id_old", "")) == str(cluster_value)]

        if sort_value == "prob_desc":
            filtered = sorted(filtered, key=lambda row: row["prob_small"], reverse=True)
        elif sort_value == "prob_asc":
            filtered = sorted(filtered, key=lambda row: row["prob_small"])
        else:
            filtered = sorted(filtered, key=lambda row: int(row["sample_id"]))

        self.filtered_samples = filtered
        self.current_index = 0
        self.status_label.setText(f"{len(self.filtered_samples)} / {len(self.samples)}")
        self.plot_current_sample()

    def on_filters_changed(self, _index):
        self.apply_filters()

    def plot_current_sample(self):
        self.ax_agc.clear()
        self.ax_origin.clear()
        self.ax_bandpass.clear()

        if not self.filtered_samples:
            self.ax_agc.set_title("No samples after filtering")
            self.canvas.draw()
            return

        sample = self.filtered_samples[self.current_index]
        sample_id = int(sample["sample_id"])
        origin = _normalize(self.signals_origin[sample_id])
        agc = _normalize(self.signals_agc[sample_id])
        agc_matched = int(sample.get("agc_matched", 0) or 0) == 1

        origin_time = int(sample.get("time", 0) or 0)
        if self.time_mode == "absolute":
            t_origin = np.arange(origin.shape[0]) + origin_time
            t_agc = np.arange(agc.shape[0]) + origin_time
        else:
            t_origin = np.arange(origin.shape[0])
            t_agc = np.arange(agc.shape[0])

        bandpassed = _bandpass_impl(
            origin,
            self.fs,
            freqmin=self.freqmin,
            freqmax=self.freqmax,
            corners=4,
            zerophase=True,
            detrend=False,
            taper=False,
        )
        bandpassed = _normalize(bandpassed)

        if agc_matched:
            self.ax_agc.plot(t_agc, agc, color="darkorange")
        else:
            self.ax_agc.text(0.5, 0.5, "No AGC match", ha="center", va="center", transform=self.ax_agc.transAxes)
        self.ax_origin.plot(t_origin, origin, color="steelblue")
        self.ax_bandpass.plot(t_origin, bandpassed, color="seagreen")

        title = (
            f"id={sample_id} pred={sample['pred_label']} prob_small={sample['prob_small']:.3f} "
            f"bucket={sample['confidence_bucket']} cluster_old={sample.get('cluster_id_old', '')} "
            f"veh={sample.get('veh_id', '')} sta={sample.get('sta_name', '')}"
        )
        self.ax_agc.set_title(f"AGC | {title}")
        self.ax_origin.set_title(f"Origin | {title}")
        self.ax_bandpass.set_title(f"Origin Bandpass ({self.freqmin}-{self.freqmax} Hz) | {title}")
        self.ax_bandpass.set_xlabel("Time index" if self.time_mode == "absolute" else "Local index")
        self.ax_agc.set_ylabel("Normalized amplitude")
        self.ax_origin.set_ylabel("Normalized amplitude")
        self.ax_bandpass.set_ylabel("Normalized amplitude")
        self.canvas.draw()

    def on_mouse_scroll(self, event):
        step = self._get_scroll_step(event)
        if step == 0:
            return

        key_state = (getattr(event, "key", "") or "").lower()
        modifiers = QApplication.keyboardModifiers()
        gui_event = getattr(event, "guiEvent", None)
        gui_modifiers = gui_event.modifiers() if gui_event is not None and hasattr(gui_event, "modifiers") else Qt.KeyboardModifier.NoModifier
        shift_pressed = (
            ("shift" in key_state)
            or bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
            or bool(gui_modifiers & Qt.KeyboardModifier.ShiftModifier)
        )
        if shift_pressed:
            if event.xdata is None:
                return
            left, right = self.ax_agc.get_xlim()
            cur_span = right - left
            if cur_span <= 0:
                return
            scale = 0.8 if step > 0 else 1.25
            new_span = cur_span * scale
            ratio = (event.xdata - left) / cur_span
            new_left = event.xdata - new_span * ratio
            new_right = new_left + new_span
            self.ax_agc.set_xlim(new_left, new_right)
            self.ax_origin.set_xlim(new_left, new_right)
            self.ax_bandpass.set_xlim(new_left, new_right)
            self.canvas.draw_idle()
            return

        if not self.filtered_samples:
            return
        self.current_index = (self.current_index - step) % len(self.filtered_samples)
        self.plot_current_sample()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_C:
            self.time_mode = "relative" if self.time_mode == "absolute" else "absolute"
            self.plot_current_sample()
        elif event.key() == Qt.Key.Key_R and self.filtered_samples:
            self.current_index = random.randrange(len(self.filtered_samples))
            self.plot_current_sample()


def main() -> None:
    args = parse_args()
    signals_origin, signals_agc, meta_rows, _ = load_paired_dataset(
        Path(args.dataset_npz),
        Path(args.meta_csv),
        mmap_mode="r",
    )
    prediction_rows = _read_predictions(Path(args.predictions_csv))

    app = QApplication(sys.argv)
    viewer = ReviewViewer(
        signals_origin=signals_origin,
        signals_agc=signals_agc,
        meta_rows=meta_rows,
        prediction_rows=prediction_rows,
        fs=args.fs,
        freqmin=args.freqmin,
        freqmax=args.freqmax,
    )
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
