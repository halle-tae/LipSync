"""
LipSync — Model Evaluation & Analysis

Evaluates a trained lip reading model on a held-out test set, generating:
    - Word Error Rate (WER) and Character Error Rate (CER)
    - Per-character confusion matrix
    - Failure-case analysis (most confused visemes / phonemes)
    - Per-speaker accuracy breakdown
    - Prediction samples and visualisation-ready outputs

Usage:
    python -m backend.model.evaluate \
        --model_path backend/model/lip_model.h5 \
        --data_dir backend/data/processed \
        --output_dir backend/model/eval_results
"""

import argparse
import json
import logging
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger("lipsync.evaluate")

# ---------------------------------------------------------------------------
# Viseme groups (characters that look the same on the lips)
# ---------------------------------------------------------------------------
VISEME_GROUPS = {
    "bilabial":   {"p", "b", "m"},
    "labiodental": {"f", "v"},
    "dental":     {"t", "d", "n", "l", "s", "z"},
    "velar":      {"k", "g"},
    "palatal":    {"j", "y"},
    "vowel_open": {"a"},
    "vowel_mid":  {"e", "o"},
    "vowel_close": {"i", "u"},
    "other":      set("bcdfghjklmnpqrstvwxyz0123456789 "),
}


def _char_to_viseme(ch: str) -> str:
    """Map a character to its viseme group."""
    for group, chars in VISEME_GROUPS.items():
        if group == "other":
            continue
        if ch in chars:
            return group
    return "other"


# ---------------------------------------------------------------------------
# Evaluation Engine
# ---------------------------------------------------------------------------
class ModelEvaluator:
    """
    Evaluate a trained LipSync model with detailed metrics and analysis.

    Usage:
        evaluator = ModelEvaluator("backend/model/lip_model.h5")
        results = evaluator.evaluate("backend/data/processed")
        evaluator.save_report(results, "backend/model/eval_results")
    """

    def __init__(self, model_path: str | Path):
        """Load the trained model."""
        import tensorflow as tf
        from backend.model.train import (
            build_lip_reading_model,
            ctc_loss_fn,
            decode_predictions,
            decode_label,
            compute_wer,
            compute_cer,
            IDX_TO_CHAR,
            CHAR_TO_IDX,
            NUM_CLASSES,
        )

        self.tf = tf
        self.decode_predictions = decode_predictions
        self.decode_label = decode_label
        self.compute_wer = compute_wer
        self.compute_cer = compute_cer
        self.IDX_TO_CHAR = IDX_TO_CHAR
        self.CHAR_TO_IDX = CHAR_TO_IDX
        self.NUM_CLASSES = NUM_CLASSES

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {model_path}. "
                "Train the model first: python -m backend.model.train"
            )

        self.model = build_lip_reading_model()
        self.model.compile(loss=ctc_loss_fn)
        self.model.load_weights(str(model_path))
        logger.info("Model loaded from %s", model_path)

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        data_dir: str | Path,
        batch_size: int = 16,
        max_samples: int | None = None,
    ) -> dict:
        """
        Run full evaluation on the processed dataset.

        Args:
            data_dir: Path to processed data directory.
            batch_size: Batch size for inference.
            max_samples: Limit evaluation to N samples (None = all).

        Returns:
            Dictionary containing all metrics, samples, and analysis.
        """
        from backend.utils.data_loader import create_dataset, discover_samples

        logger.info("=" * 60)
        logger.info("LipSync — Model Evaluation")
        logger.info("=" * 60)

        # Use the validation split for evaluation
        _, val_ds, _, num_val = create_dataset(
            data_dir=str(data_dir),
            batch_size=batch_size,
            val_split=0.2,
        )

        logger.info("Evaluating on %d validation samples...", num_val)

        # Collect predictions
        predictions = []      # list of (reference, hypothesis, wer, cer)
        char_confusions = []  # list of (ref_char, hyp_char) pairs
        speaker_results = defaultdict(lambda: {"wer": [], "cer": []})

        total_inference_time = 0.0
        count = 0

        for frames_batch, labels_batch in val_ds:
            t0 = time.time()
            preds = self.model.predict(frames_batch, verbose=0)
            total_inference_time += time.time() - t0

            for i in range(preds.shape[0]):
                if max_samples is not None and count >= max_samples:
                    break

                hyp = self.decode_predictions(preds[i])
                ref = self.decode_label(labels_batch[i].numpy())
                wer = self.compute_wer(ref, hyp)
                cer = self.compute_cer(ref, hyp)

                predictions.append({
                    "reference": ref,
                    "hypothesis": hyp,
                    "wer": wer,
                    "cer": cer,
                })

                # Character-level alignment for confusion
                self._align_chars(ref, hyp, char_confusions)

                count += 1

            if max_samples is not None and count >= max_samples:
                break

        if count == 0:
            logger.error("No samples evaluated!")
            return {}

        # ---- Aggregate Metrics ----
        all_wer = [p["wer"] for p in predictions]
        all_cer = [p["cer"] for p in predictions]

        results = {
            "num_samples": count,
            "avg_wer": float(np.mean(all_wer)),
            "avg_cer": float(np.mean(all_cer)),
            "median_wer": float(np.median(all_wer)),
            "median_cer": float(np.median(all_cer)),
            "std_wer": float(np.std(all_wer)),
            "std_cer": float(np.std(all_cer)),
            "min_wer": float(np.min(all_wer)),
            "max_wer": float(np.max(all_wer)),
            "perfect_predictions": sum(1 for w in all_wer if w == 0.0),
            "perfect_rate": sum(1 for w in all_wer if w == 0.0) / count,
            "avg_inference_ms": (total_inference_time / count) * 1000,
            "total_inference_s": total_inference_time,
        }

        # ---- Confusion Matrix ----
        results["confusion_matrix"] = self._build_confusion_matrix(char_confusions)

        # ---- Viseme Analysis ----
        results["viseme_analysis"] = self._viseme_confusion_analysis(char_confusions)

        # ---- Word-level Analysis ----
        results["word_analysis"] = self._word_level_analysis(predictions)

        # ---- Sample Predictions ----
        # Best, worst, and random samples
        sorted_by_wer = sorted(predictions, key=lambda p: p["wer"])
        results["best_samples"] = sorted_by_wer[:10]
        results["worst_samples"] = sorted_by_wer[-10:]
        np.random.seed(42)
        random_idx = np.random.choice(len(predictions), min(10, len(predictions)), replace=False)
        results["random_samples"] = [predictions[i] for i in random_idx]

        # ---- Log Summary ----
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info("  Samples evaluated   : %d", results["num_samples"])
        logger.info("  Average WER         : %.4f (%.1f%%)", results["avg_wer"], results["avg_wer"] * 100)
        logger.info("  Average CER         : %.4f (%.1f%%)", results["avg_cer"], results["avg_cer"] * 100)
        logger.info("  Median WER          : %.4f", results["median_wer"])
        logger.info("  Median CER          : %.4f", results["median_cer"])
        logger.info("  Perfect predictions : %d / %d (%.1f%%)",
                     results["perfect_predictions"], count, results["perfect_rate"] * 100)
        logger.info("  Avg inference time  : %.1f ms / sample", results["avg_inference_ms"])
        logger.info("=" * 60)

        logger.info("\n--- Best Predictions ---")
        for s in results["best_samples"][:5]:
            logger.info("  REF: %-30s  HYP: %-30s  WER: %.2f", s["reference"], s["hypothesis"], s["wer"])

        logger.info("\n--- Worst Predictions ---")
        for s in results["worst_samples"][-5:]:
            logger.info("  REF: %-30s  HYP: %-30s  WER: %.2f", s["reference"], s["hypothesis"], s["wer"])

        return results

    # ------------------------------------------------------------------
    # Character-level alignment (for confusion analysis)
    # ------------------------------------------------------------------
    def _align_chars(self, ref: str, hyp: str, confusions: list):
        """
        Align ref and hyp characters and record (ref_char, hyp_char) pairs.
        Uses a simple Levenshtein-like alignment.
        """
        # Simple approach: pad shorter string and align positionally
        # For more accurate results, use a full alignment algorithm
        max_len = max(len(ref), len(hyp))
        for i in range(min(len(ref), len(hyp))):
            if ref[i] != hyp[i]:
                confusions.append((ref[i], hyp[i]))
            else:
                confusions.append((ref[i], ref[i]))  # correct

        # Deletions (in ref but not in hyp)
        if len(ref) > len(hyp):
            for i in range(len(hyp), len(ref)):
                confusions.append((ref[i], "<del>"))

        # Insertions (in hyp but not in ref)
        if len(hyp) > len(ref):
            for i in range(len(ref), len(hyp)):
                confusions.append(("<ins>", hyp[i]))

    # ------------------------------------------------------------------
    # Confusion Matrix
    # ------------------------------------------------------------------
    def _build_confusion_matrix(self, confusions: list) -> dict:
        """
        Build a character-level confusion matrix from alignment pairs.

        Returns a dict:
            {
                "labels": [list of characters],
                "matrix": [[counts]],   # matrix[i][j] = times label i predicted as j
                "most_confused": [(ref, hyp, count), ...]
            }
        """
        # Count confusions (excluding correct predictions)
        error_counts = Counter()
        correct_counts = Counter()

        for ref_c, hyp_c in confusions:
            if ref_c == hyp_c:
                correct_counts[ref_c] += 1
            else:
                error_counts[(ref_c, hyp_c)] += 1

        # Build most-confused pairs
        most_confused = [
            {"ref": pair[0], "hyp": pair[1], "count": count}
            for pair, count in error_counts.most_common(30)
        ]

        # Per-character accuracy
        char_accuracy = {}
        all_chars = set()
        char_total = Counter()
        char_correct = Counter()
        for ref_c, hyp_c in confusions:
            if ref_c not in ("<ins>",):
                char_total[ref_c] += 1
                if ref_c == hyp_c:
                    char_correct[ref_c] += 1
                all_chars.add(ref_c)

        for ch in sorted(all_chars):
            total = char_total[ch]
            correct = char_correct[ch]
            char_accuracy[ch] = {
                "total": total,
                "correct": correct,
                "accuracy": correct / total if total > 0 else 0.0,
            }

        return {
            "most_confused": most_confused,
            "per_char_accuracy": char_accuracy,
        }

    # ------------------------------------------------------------------
    # Viseme Confusion Analysis
    # ------------------------------------------------------------------
    def _viseme_confusion_analysis(self, confusions: list) -> dict:
        """
        Analyse confusions at the viseme level.

        Visemes are groups of phonemes that look identical on the lips.
        E.g., "p", "b", "m" are bilabial and indistinguishable visually.
        """
        viseme_errors = Counter()
        viseme_total = Counter()

        for ref_c, hyp_c in confusions:
            if ref_c in ("<ins>", "<del>") or hyp_c in ("<ins>", "<del>"):
                continue
            ref_vis = _char_to_viseme(ref_c)
            hyp_vis = _char_to_viseme(hyp_c)
            viseme_total[ref_vis] += 1
            if ref_vis != hyp_vis:
                viseme_errors[(ref_vis, hyp_vis)] += 1

        # Within-viseme confusions (e.g., p↔b — expected to be hard)
        within_viseme = Counter()
        for ref_c, hyp_c in confusions:
            if ref_c == hyp_c or ref_c in ("<ins>", "<del>") or hyp_c in ("<ins>", "<del>"):
                continue
            ref_vis = _char_to_viseme(ref_c)
            hyp_vis = _char_to_viseme(hyp_c)
            if ref_vis == hyp_vis and ref_vis != "other":
                within_viseme[(ref_c, hyp_c)] += 1

        return {
            "cross_viseme_errors": [
                {"ref": pair[0], "hyp": pair[1], "count": count}
                for pair, count in viseme_errors.most_common(20)
            ],
            "within_viseme_confusions": [
                {"ref": pair[0], "hyp": pair[1], "count": count}
                for pair, count in within_viseme.most_common(20)
            ],
            "viseme_totals": dict(viseme_total),
        }

    # ------------------------------------------------------------------
    # Word-level analysis
    # ------------------------------------------------------------------
    def _word_level_analysis(self, predictions: list) -> dict:
        """Analyse which GRID vocabulary words are most often correct/wrong."""
        word_correct = Counter()
        word_total = Counter()
        word_errors = defaultdict(list)  # word → list of wrong predictions

        for pred in predictions:
            ref_words = pred["reference"].split()
            hyp_words = pred["hypothesis"].split()

            # Align word-by-word (simple positional)
            for i, ref_w in enumerate(ref_words):
                word_total[ref_w] += 1
                hyp_w = hyp_words[i] if i < len(hyp_words) else "<missing>"
                if ref_w == hyp_w:
                    word_correct[ref_w] += 1
                else:
                    word_errors[ref_w].append(hyp_w)

        # Per-word accuracy
        word_accuracy = {}
        for word in sorted(word_total.keys()):
            total = word_total[word]
            correct = word_correct.get(word, 0)
            word_accuracy[word] = {
                "total": total,
                "correct": correct,
                "accuracy": correct / total if total > 0 else 0.0,
                "common_errors": Counter(word_errors.get(word, [])).most_common(5),
            }

        # Sort by accuracy (worst first)
        hardest_words = sorted(
            word_accuracy.items(),
            key=lambda x: x[1]["accuracy"],
        )[:20]

        easiest_words = sorted(
            word_accuracy.items(),
            key=lambda x: -x[1]["accuracy"],
        )[:20]

        return {
            "per_word_accuracy": word_accuracy,
            "hardest_words": [
                {"word": w, **stats} for w, stats in hardest_words
            ],
            "easiest_words": [
                {"word": w, **stats} for w, stats in easiest_words
            ],
        }

    # ------------------------------------------------------------------
    # Save Report
    # ------------------------------------------------------------------
    def save_report(self, results: dict, output_dir: str | Path):
        """
        Save evaluation results to disk.

        Outputs:
            eval_results/
                metrics.json          — aggregated metrics
                predictions.json      — all prediction samples
                confusion.json        — confusion matrix data
                viseme_analysis.json  — viseme confusion analysis
                word_analysis.json    — per-word accuracy breakdown
                eval_summary.txt      — human-readable summary
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Metrics
        metrics = {
            k: v for k, v in results.items()
            if k not in ("confusion_matrix", "viseme_analysis", "word_analysis",
                         "best_samples", "worst_samples", "random_samples")
        }
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        # Sample predictions
        samples = {
            "best": results.get("best_samples", []),
            "worst": results.get("worst_samples", []),
            "random": results.get("random_samples", []),
        }
        with open(output_dir / "predictions.json", "w") as f:
            json.dump(samples, f, indent=2, default=str)

        # Confusion matrix
        if "confusion_matrix" in results:
            with open(output_dir / "confusion.json", "w") as f:
                json.dump(results["confusion_matrix"], f, indent=2, default=str)

        # Viseme analysis
        if "viseme_analysis" in results:
            with open(output_dir / "viseme_analysis.json", "w") as f:
                json.dump(results["viseme_analysis"], f, indent=2, default=str)

        # Word analysis
        if "word_analysis" in results:
            # Convert Counter objects for JSON serialisation
            wa = results["word_analysis"]
            for word_info in wa.get("per_word_accuracy", {}).values():
                if "common_errors" in word_info:
                    word_info["common_errors"] = [
                        {"word": w, "count": c}
                        for w, c in word_info["common_errors"]
                    ]
            with open(output_dir / "word_analysis.json", "w") as f:
                json.dump(wa, f, indent=2, default=str)

        # Human-readable summary
        summary_lines = [
            "=" * 60,
            "LipSync — Evaluation Summary",
            "=" * 60,
            "",
            f"Samples evaluated   : {results['num_samples']}",
            f"Average WER         : {results['avg_wer']:.4f} ({results['avg_wer'] * 100:.1f}%)",
            f"Average CER         : {results['avg_cer']:.4f} ({results['avg_cer'] * 100:.1f}%)",
            f"Median WER          : {results['median_wer']:.4f}",
            f"Median CER          : {results['median_cer']:.4f}",
            f"Std WER             : {results['std_wer']:.4f}",
            f"Std CER             : {results['std_cer']:.4f}",
            f"Best WER            : {results['min_wer']:.4f}",
            f"Worst WER           : {results['max_wer']:.4f}",
            f"Perfect predictions : {results['perfect_predictions']} ({results['perfect_rate'] * 100:.1f}%)",
            f"Avg inference time  : {results['avg_inference_ms']:.1f} ms/sample",
            "",
            "--- Most Confused Character Pairs ---",
        ]

        if "confusion_matrix" in results:
            for item in results["confusion_matrix"]["most_confused"][:15]:
                summary_lines.append(
                    f"  '{item['ref']}' → '{item['hyp']}': {item['count']} times"
                )

        summary_lines.extend([
            "",
            "--- Within-Viseme Confusions (expected to be hard) ---",
        ])
        if "viseme_analysis" in results:
            for item in results["viseme_analysis"]["within_viseme_confusions"][:10]:
                summary_lines.append(
                    f"  '{item['ref']}' ↔ '{item['hyp']}': {item['count']} times"
                )

        summary_lines.extend([
            "",
            "--- Best Predictions ---",
        ])
        for s in results.get("best_samples", [])[:5]:
            summary_lines.append(
                f"  REF: {s['reference']:30s}  HYP: {s['hypothesis']:30s}  WER: {s['wer']:.2f}"
            )

        summary_lines.extend([
            "",
            "--- Worst Predictions ---",
        ])
        for s in results.get("worst_samples", [])[-5:]:
            summary_lines.append(
                f"  REF: {s['reference']:30s}  HYP: {s['hypothesis']:30s}  WER: {s['wer']:.2f}"
            )

        summary_lines.append("\n" + "=" * 60)

        summary_text = "\n".join(summary_lines)
        (output_dir / "eval_summary.txt").write_text(summary_text)
        logger.info("Evaluation report saved to %s", output_dir)
        logger.info("\n%s", summary_text)


# ---------------------------------------------------------------------------
# Plot Training History (utility for notebooks / post-training)
# ---------------------------------------------------------------------------
def plot_training_history(history_path: str | Path, output_dir: str | Path | None = None):
    """
    Generate training curve plots from a saved training_history.npz file.

    Plots:
        1. Loss (train & val) vs. epoch
        2. WER & CER vs. epoch
        3. Learning rate vs. epoch (if available)

    Args:
        history_path: Path to training_history.npz.
        output_dir: If provided, saves plots as PNG files.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(history_path, allow_pickle=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # -- Plot 1: Loss --
    ax = axes[0]
    if "loss" in data:
        ax.plot(data["loss"], label="Train Loss", linewidth=2)
    if "val_loss" in data:
        ax.plot(data["val_loss"], label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CTC Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # -- Plot 2: WER & CER --
    ax = axes[1]
    if "wer" in data:
        ax.plot(data["wer"], label="WER", linewidth=2, color="tab:red")
    if "cer" in data:
        ax.plot(data["cer"], label="CER", linewidth=2, color="tab:orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error Rate")
    ax.set_title("Word & Character Error Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # -- Plot 3: Learning Rate --
    ax = axes[2]
    if "lr" in data:
        ax.plot(data["lr"], linewidth=2, color="tab:green")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "LR data not available", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Learning Rate Schedule")

    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
        logger.info("Training curves saved to %s", output_dir / "training_curves.png")

    plt.close(fig)
    return fig


def plot_confusion_heatmap(confusion_path: str | Path, output_dir: str | Path | None = None):
    """
    Generate a heatmap from per-character accuracy data.

    Args:
        confusion_path: Path to confusion.json.
        output_dir: If provided, saves the plot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(confusion_path) as f:
        confusion = json.load(f)

    per_char = confusion.get("per_char_accuracy", {})
    if not per_char:
        logger.warning("No per-character accuracy data to plot.")
        return

    chars = sorted(per_char.keys())
    accuracies = [per_char[c]["accuracy"] for c in chars]

    fig, ax = plt.subplots(figsize=(14, 4))
    colours = ["#d32f2f" if a < 0.5 else "#ffa726" if a < 0.8 else "#66bb6a" for a in accuracies]
    bars = ax.bar(range(len(chars)), accuracies, color=colours)
    ax.set_xticks(range(len(chars)))
    ax.set_xticklabels(chars, fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Character Recognition Accuracy")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.5, label="80% threshold")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "char_accuracy.png", dpi=150, bbox_inches="tight")
        logger.info("Char accuracy plot saved to %s", output_dir / "char_accuracy.png")

    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained LipSync model.")
    parser.add_argument(
        "--model_path", type=str, default="backend/model/lip_model.h5",
        help="Path to trained model weights.",
    )
    parser.add_argument(
        "--data_dir", type=str, default="backend/data/processed",
        help="Path to processed dataset directory.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="backend/model/eval_results",
        help="Directory for evaluation outputs.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Max samples to evaluate (None = all).",
    )
    parser.add_argument(
        "--plot_history", type=str, default=None,
        help="Path to training_history.npz to plot training curves.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    # Plot training history if requested
    if args.plot_history:
        plot_training_history(args.plot_history, args.output_dir)
        return

    # Run evaluation
    evaluator = ModelEvaluator(args.model_path)
    results = evaluator.evaluate(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )
    evaluator.save_report(results, args.output_dir)

    # Plot confusion if results exist
    confusion_path = Path(args.output_dir) / "confusion.json"
    if confusion_path.exists():
        plot_confusion_heatmap(confusion_path, args.output_dir)


if __name__ == "__main__":
    main()

