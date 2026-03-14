"""
LipSync — Model Optimization for Real-Time Inference

Profiles model inference time and provides multiple optimization paths:
    1. TensorFlow Lite conversion (float32 + INT8 quantized)
    2. Reduced model variant (fewer parameters)
    3. Inference profiling and benchmarking

Target: < 150ms inference latency per 75-frame sequence on a standard
laptop CPU/GPU.

Usage:
    # Profile the model
    python -m backend.model.optimize --action profile \
        --model_path backend/model/lip_model.h5

    # Convert to TFLite
    python -m backend.model.optimize --action tflite \
        --model_path backend/model/lip_model.h5 \
        --output_dir backend/model

    # Build a smaller model variant
    python -m backend.model.optimize --action small \
        --output_dir backend/model

    # Run full benchmark (Keras vs TFLite vs TFLite-quantized)
    python -m backend.model.optimize --action benchmark \
        --model_path backend/model/lip_model.h5
"""

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("lipsync.optimize")


# ---------------------------------------------------------------------------
# Constants (must match train.py)
# ---------------------------------------------------------------------------
MAX_FRAMES = 75
IMG_HEIGHT = 50
IMG_WIDTH = 100
IMG_CHANNELS = 1


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------
def profile_model(
    model_path: str | Path,
    num_runs: int = 50,
    warmup_runs: int = 5,
) -> dict:
    """
    Profile inference latency of the trained Keras model.

    Args:
        model_path: Path to saved model weights (.h5).
        num_runs: Number of inference runs to average.
        warmup_runs: Number of warm-up runs before timing.

    Returns:
        Dict with timing statistics (in milliseconds).
    """
    import tensorflow as tf
    from backend.model.train import build_lip_reading_model, ctc_loss_fn

    logger.info("=" * 60)
    logger.info("Model Inference Profiling")
    logger.info("=" * 60)

    # Load model
    model = build_lip_reading_model()
    model.compile(loss=ctc_loss_fn)

    model_path = Path(model_path)
    if model_path.exists():
        model.load_weights(str(model_path))
        logger.info("Loaded weights from %s", model_path)
    else:
        logger.warning("Weights not found at %s — profiling with random weights", model_path)

    # Create dummy input
    dummy_input = np.random.rand(1, MAX_FRAMES, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS).astype(np.float32)

    # Warm-up
    logger.info("Running %d warm-up iterations...", warmup_runs)
    for _ in range(warmup_runs):
        model.predict(dummy_input, verbose=0)

    # Timed runs
    logger.info("Running %d timed iterations...", num_runs)
    timings = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        model.predict(dummy_input, verbose=0)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000)  # ms

    timings = np.array(timings)
    results = {
        "mean_ms": float(np.mean(timings)),
        "median_ms": float(np.median(timings)),
        "std_ms": float(np.std(timings)),
        "min_ms": float(np.min(timings)),
        "max_ms": float(np.max(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
        "p99_ms": float(np.percentile(timings, 99)),
        "num_runs": num_runs,
        "model_params": model.count_params(),
    }

    # Device info
    gpus = tf.config.list_physical_devices("GPU")
    results["device"] = gpus[0].name if gpus else "CPU"

    logger.info("\nKeras Model Profiling Results:")
    logger.info("  Mean latency   : %.1f ms", results["mean_ms"])
    logger.info("  Median latency : %.1f ms", results["median_ms"])
    logger.info("  Std dev        : %.1f ms", results["std_ms"])
    logger.info("  Min latency    : %.1f ms", results["min_ms"])
    logger.info("  Max latency    : %.1f ms", results["max_ms"])
    logger.info("  P95 latency    : %.1f ms", results["p95_ms"])
    logger.info("  P99 latency    : %.1f ms", results["p99_ms"])
    logger.info("  Parameters     : %s", f"{results['model_params']:,}")
    logger.info("  Device         : %s", results["device"])

    target_met = results["median_ms"] < 150
    if target_met:
        logger.info("\n  ✅ Target met: median latency < 150ms")
    else:
        logger.info("\n  ⚠️  Target NOT met: median latency >= 150ms")
        logger.info("     Consider: TFLite conversion, quantization, or a smaller model.")

    return results


# ---------------------------------------------------------------------------
# TFLite Conversion
# ---------------------------------------------------------------------------
def convert_to_tflite(
    model_path: str | Path,
    output_dir: str | Path,
    quantize: bool = True,
) -> dict:
    """
    Convert the Keras model to TensorFlow Lite format.

    Creates two variants:
        1. lip_model.tflite          — float32 (full precision)
        2. lip_model_quantized.tflite — INT8 post-training quantization

    Args:
        model_path: Path to saved Keras model weights (.h5).
        output_dir: Directory for output .tflite files.
        quantize: Whether to also produce a quantized model.

    Returns:
        Dict with file sizes and conversion status.
    """
    import tensorflow as tf
    from backend.model.train import build_lip_reading_model, ctc_loss_fn

    logger.info("=" * 60)
    logger.info("TFLite Conversion")
    logger.info("=" * 60)

    # Build and load model
    model = build_lip_reading_model()
    model.compile(loss=ctc_loss_fn)

    model_path = Path(model_path)
    if model_path.exists():
        model.load_weights(str(model_path))
        logger.info("Loaded weights from %s", model_path)
    else:
        logger.warning("Weights not found — converting with random weights for testing")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # --- Float32 TFLite ---
    logger.info("\nConverting to float32 TFLite...")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,  # For ops not natively supported
        ]
        converter._experimental_lower_tensor_list_ops = False
        tflite_model = converter.convert()

        tflite_path = output_dir / "lip_model.tflite"
        tflite_path.write_bytes(tflite_model)
        results["float32_path"] = str(tflite_path)
        results["float32_size_mb"] = len(tflite_model) / (1024 * 1024)
        logger.info("  ✅ Saved: %s (%.1f MB)", tflite_path, results["float32_size_mb"])
    except Exception as e:
        logger.error("  ❌ Float32 conversion failed: %s", e)
        results["float32_error"] = str(e)

    # --- INT8 Quantized TFLite ---
    if quantize:
        logger.info("\nConverting to INT8 quantized TFLite...")
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            converter._experimental_lower_tensor_list_ops = False

            # Representative dataset for calibration
            def representative_dataset():
                for _ in range(100):
                    data = np.random.rand(
                        1, MAX_FRAMES, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS
                    ).astype(np.float32)
                    yield [data]

            converter.representative_dataset = representative_dataset

            quant_model = converter.convert()

            quant_path = output_dir / "lip_model_quantized.tflite"
            quant_path.write_bytes(quant_model)
            results["quantized_path"] = str(quant_path)
            results["quantized_size_mb"] = len(quant_model) / (1024 * 1024)
            logger.info("  ✅ Saved: %s (%.1f MB)", quant_path, results["quantized_size_mb"])

            if "float32_size_mb" in results:
                ratio = results["quantized_size_mb"] / results["float32_size_mb"]
                logger.info(
                    "  Compression: %.1fx (%.1f MB → %.1f MB)",
                    1.0 / ratio, results["float32_size_mb"], results["quantized_size_mb"],
                )
        except Exception as e:
            logger.error("  ❌ Quantized conversion failed: %s", e)
            results["quantized_error"] = str(e)

    return results


# ---------------------------------------------------------------------------
# TFLite Inference Benchmarking
# ---------------------------------------------------------------------------
def benchmark_tflite(
    tflite_path: str | Path,
    num_runs: int = 50,
    warmup_runs: int = 5,
) -> dict:
    """
    Benchmark inference latency of a TFLite model.

    Args:
        tflite_path: Path to .tflite model file.
        num_runs: Number of timed inference runs.
        warmup_runs: Number of warm-up runs.

    Returns:
        Dict with timing statistics (ms).
    """
    import tensorflow as tf

    tflite_path = Path(tflite_path)
    if not tflite_path.exists():
        raise FileNotFoundError(f"TFLite model not found at {tflite_path}")

    logger.info("Benchmarking TFLite model: %s", tflite_path)

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]
    logger.info("  Input shape: %s", input_shape)

    dummy_input = np.random.rand(*input_shape).astype(np.float32)

    # Warm-up
    for _ in range(warmup_runs):
        interpreter.set_tensor(input_details[0]["index"], dummy_input)
        interpreter.invoke()

    # Timed runs
    timings = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], dummy_input)
        interpreter.invoke()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000)

    timings_arr = np.array(timings)
    file_size_mb = tflite_path.stat().st_size / (1024 * 1024)

    results = {
        "model": str(tflite_path.name),
        "mean_ms": float(np.mean(timings_arr)),
        "median_ms": float(np.median(timings_arr)),
        "std_ms": float(np.std(timings_arr)),
        "min_ms": float(np.min(timings_arr)),
        "max_ms": float(np.max(timings_arr)),
        "p95_ms": float(np.percentile(timings_arr, 95)),
        "file_size_mb": file_size_mb,
    }

    logger.info("  Mean: %.1f ms | Median: %.1f ms | P95: %.1f ms | Size: %.1f MB",
                results["mean_ms"], results["median_ms"], results["p95_ms"], file_size_mb)

    return results


# ---------------------------------------------------------------------------
# Smaller Model Variant
# ---------------------------------------------------------------------------
def build_small_model(save_path: str | Path | None = None):
    """
    Build a smaller variant of the LipSync model for faster inference.

    Reduced architecture:
        - Fewer CNN filters (16, 32, 48)
        - Smaller LSTM hidden size (128 units)
        - Lower dropout

    Suitable for CPU-only inference or resource-constrained environments.
    """
    import tensorflow as tf
    from backend.model.train import NUM_CLASSES

    inputs = tf.keras.Input(
        shape=(MAX_FRAMES, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
        name="lip_frames",
    )

    # Lightweight CNN
    x = tf.keras.layers.Conv3D(16, (3, 5, 5), padding="same", kernel_initializer="he_normal")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool3D((1, 2, 2))(x)

    x = tf.keras.layers.Conv3D(32, (3, 5, 5), padding="same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool3D((1, 2, 2))(x)

    x = tf.keras.layers.Conv3D(48, (3, 3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool3D((1, 2, 2))(x)

    # Collapse spatial → temporal
    time_steps = x.shape[1]
    feature_dim = x.shape[2] * x.shape[3] * x.shape[4]
    x = tf.keras.layers.Reshape((time_steps, feature_dim))(x)

    # Lighter BiLSTM
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.3)
    )(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.3)
    )(x)

    x = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="ctc_output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=x, name="LipSyncSmall")
    model.summary(print_fn=logger.info)

    logger.info("Small model parameters: %s", f"{model.count_params():,}")

    if save_path:
        save_path = Path(save_path)
        # Save the architecture config (weights will be random until training)
        config_path = save_path.parent / "lip_model_small_config.json"
        config_path.write_text(model.to_json())
        logger.info("Small model config saved to %s", config_path)

    return model


# ---------------------------------------------------------------------------
# Full Benchmark (compare all variants)
# ---------------------------------------------------------------------------
def full_benchmark(model_path: str | Path, output_dir: str | Path):
    """
    Run a comprehensive benchmark comparing all model variants:
        1. Keras model (original)
        2. TFLite float32
        3. TFLite INT8 quantized
        4. Small model variant

    Results are printed and saved to benchmark_results.json.
    """
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # 1. Profile Keras model
    logger.info("\n" + "=" * 60)
    logger.info("1/4 — Profiling Keras model")
    logger.info("=" * 60)
    try:
        all_results["keras"] = profile_model(model_path, num_runs=30)
    except Exception as e:
        logger.error("Keras profiling failed: %s", e)
        all_results["keras"] = {"error": str(e)}

    # 2. Convert and benchmark TFLite
    logger.info("\n" + "=" * 60)
    logger.info("2/4 — TFLite conversion")
    logger.info("=" * 60)
    try:
        tflite_results = convert_to_tflite(model_path, output_dir, quantize=True)
        all_results["tflite_conversion"] = tflite_results

        # Benchmark float32 TFLite
        if "float32_path" in tflite_results:
            logger.info("\n--- TFLite float32 benchmark ---")
            all_results["tflite_float32"] = benchmark_tflite(
                tflite_results["float32_path"], num_runs=30
            )

        # Benchmark quantized TFLite
        if "quantized_path" in tflite_results:
            logger.info("\n--- TFLite INT8 quantized benchmark ---")
            all_results["tflite_quantized"] = benchmark_tflite(
                tflite_results["quantized_path"], num_runs=30
            )
    except Exception as e:
        logger.error("TFLite benchmark failed: %s", e)
        all_results["tflite_error"] = str(e)

    # 3. Profile small model
    logger.info("\n" + "=" * 60)
    logger.info("3/4 — Small model variant")
    logger.info("=" * 60)
    try:
        small_model = build_small_model()
        # Quick profile
        dummy = np.random.rand(1, MAX_FRAMES, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS).astype(np.float32)
        for _ in range(3):
            small_model.predict(dummy, verbose=0)
        times = []
        for _ in range(30):
            t0 = time.perf_counter()
            small_model.predict(dummy, verbose=0)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        all_results["small_model"] = {
            "mean_ms": float(np.mean(times)),
            "median_ms": float(np.median(times)),
            "params": small_model.count_params(),
        }
    except Exception as e:
        logger.error("Small model profiling failed: %s", e)

    # 4. Summary comparison
    logger.info("\n" + "=" * 60)
    logger.info("4/4 — BENCHMARK SUMMARY")
    logger.info("=" * 60)
    logger.info("%-25s  %10s  %10s  %12s", "Variant", "Median(ms)", "Mean(ms)", "Size/Params")
    logger.info("-" * 60)

    for variant, data in all_results.items():
        if isinstance(data, dict) and "median_ms" in data:
            size_info = ""
            if "file_size_mb" in data:
                size_info = f"{data['file_size_mb']:.1f} MB"
            elif "model_params" in data:
                size_info = f"{data['model_params']:,} params"
            elif "params" in data:
                size_info = f"{data['params']:,} params"
            logger.info(
                "%-25s  %10.1f  %10.1f  %12s",
                variant, data["median_ms"], data["mean_ms"], size_info,
            )

    # Save results
    results_path = output_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("\nBenchmark results saved to %s", results_path)

    return all_results


# ---------------------------------------------------------------------------
# TFLite Predictor (for use in inference engine)
# ---------------------------------------------------------------------------
class TFLitePredictor:
    """
    Run inference using a TFLite model for optimized real-time performance.

    Usage:
        predictor = TFLitePredictor("backend/model/lip_model.tflite")
        text, confidence = predictor.predict(frames)
    """

    def __init__(self, model_path: str | Path):
        import tensorflow as tf

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"TFLite model not found at {model_path}")

        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        logger.info("TFLite model loaded from %s", model_path)
        logger.info("  Input shape: %s", self.input_details[0]["shape"])
        logger.info("  Output shape: %s", self.output_details[0]["shape"])

    def predict(self, frames: np.ndarray) -> tuple[str, float]:
        """
        Run inference on a sequence of lip ROI frames.

        Args:
            frames: Lip ROI frame sequence, shape (frames, H, W) or (frames, H, W, 1).

        Returns:
            (predicted_text, confidence_score)
        """
        from backend.model.train import decode_predictions

        # Preprocess
        if frames.ndim == 3:
            frames = frames[..., np.newaxis]
        if frames.max() > 1.0:
            frames = frames.astype(np.float32) / 255.0

        # Pad/truncate
        num_frames = frames.shape[0]
        if num_frames < MAX_FRAMES:
            pad = np.zeros(
                (MAX_FRAMES - num_frames, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                dtype=np.float32,
            )
            frames = np.concatenate([frames, pad], axis=0)
        elif num_frames > MAX_FRAMES:
            frames = frames[:MAX_FRAMES]

        # Add batch dim
        input_data = frames[np.newaxis, ...].astype(np.float32)

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])

        pred = output[0]  # (time_steps, classes)
        text = decode_predictions(pred)
        confidence = float(np.mean(np.max(pred, axis=-1)))

        return text, confidence


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="LipSync model optimization and benchmarking."
    )
    parser.add_argument(
        "--action",
        choices=["profile", "tflite", "small", "benchmark"],
        default="benchmark",
        help="Action to perform.",
    )
    parser.add_argument(
        "--model_path", type=str, default="backend/model/lip_model.h5",
        help="Path to trained Keras model weights.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="backend/model",
        help="Output directory for optimized models.",
    )
    parser.add_argument(
        "--num_runs", type=int, default=50,
        help="Number of inference runs for profiling.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    if args.action == "profile":
        profile_model(args.model_path, num_runs=args.num_runs)
    elif args.action == "tflite":
        convert_to_tflite(args.model_path, args.output_dir, quantize=True)
    elif args.action == "small":
        build_small_model(save_path=Path(args.output_dir) / "lip_model_small.h5")
    elif args.action == "benchmark":
        full_benchmark(args.model_path, args.output_dir)


if __name__ == "__main__":
    main()

