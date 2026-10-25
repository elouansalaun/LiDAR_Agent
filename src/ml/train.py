"""Train and evaluate the Random Forest point classifier.

Two experiments are run on the same spatial train/test split:
    baseline  - the 5 features listed in the roadmap
    geometric - baseline + 4 neighbourhood-shape features
"""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src import config
from src.lidar import features
from src.ml import dataset, evaluate

RF_PARAMS = dict(
    n_estimators=100,
    max_depth=18,          # limits model size and over-fitting
    min_samples_leaf=5,
    class_weight="balanced_subsample",  # buildings are rarer than ground/vegetation
    n_jobs=-1,
    random_state=0,
)

EXPERIMENTS = {
    "baseline": features.BASELINE_FEATURES,
    "geometric": features.ALL_FEATURES,
}

DOCS_IMAGES = config.PROJECT_ROOT / "docs" / "images"


def run_experiment(name: str, feature_names: list[str], train: dataset.Samples, test: dataset.Samples) -> tuple[RandomForestClassifier, dict]:
    columns = [train.feature_names.index(f) for f in feature_names]

    model = RandomForestClassifier(**RF_PARAMS)
    start = time.perf_counter()
    model.fit(train.X[:, columns], train.y)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    predictions = model.predict(test.X[:, columns])
    predict_time = time.perf_counter() - start

    metrics = evaluate.compute_metrics(test.y, predictions)
    metrics["features"] = feature_names
    metrics["train_time_s"] = round(train_time, 1)
    metrics["inference_us_per_point"] = round(1e6 * predict_time / len(test.y), 2)
    metrics["feature_importance"] = {
        f: round(float(v), 4) for f, v in sorted(zip(feature_names, model.feature_importances_), key=lambda p: -p[1])
    }

    print(f"\n=== {name} ({len(feature_names)} features) ===")
    print(evaluate.metrics_table(metrics))
    print(f"accuracy={metrics['overall_accuracy']:.3f}  train={train_time:.0f}s")
    evaluate.plot_confusion_matrix(metrics, DOCS_IMAGES / f"confusion_{name}.png", title=f"Random Forest - {name}")
    return model, metrics


def main() -> None:
    train, test = dataset.train_and_test_samples()
    print(f"\n{len(train.y):,} training points, {len(test.y):,} test points")
    for label, name in config.CLASS_NAMES.items():
        print(f"  {name:<11} train {np.mean(train.y == label):6.1%}   test {np.mean(test.y == label):6.1%}")

    results = {}
    models = {}
    for name, feature_names in EXPERIMENTS.items():
        models[name], results[name] = run_experiment(name, feature_names, train, test)

    best = max(results, key=lambda name: results[name]["mean_iou"])
    print(f"\nBest experiment: {best} (mIoU {results[best]['mean_iou']:.3f})")

    config.MODELS_DIR.mkdir(exist_ok=True)
    bundle = {
        "model": models[best],
        "features": EXPERIMENTS[best],
        "classes": config.CLASS_NAMES,
        "experiment": best,
    }
    joblib.dump(bundle, config.MODEL_PATH, compress=3)
    results["selected"] = best
    config.MODEL_METRICS_PATH.write_text(json.dumps(results, indent=2))
    print(f"Saved model to {config.MODEL_PATH} and metrics to {config.MODEL_METRICS_PATH}")


if __name__ == "__main__":
    main()
