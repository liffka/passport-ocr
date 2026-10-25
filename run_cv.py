import argparse
import hashlib
import json
from pathlib import Path
import statistics

from cv_tools import build_model, evaluate_cached, grouped_folds, grouped_metrics, passport_key, read_data
from postprocess import ROOT, Processor
from validate import summarize


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n")


def save_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8", newline="\n")


def evaluate_fold(documents, labels, assignment, fold, policy):
    training = {key: row for key, row in documents.items() if assignment[key] != fold}
    validation = {key: row for key, row in documents.items() if assignment[key] == fold}
    training_labels = {key: labels[key] for key in training}
    training_groups = {passport_key(labels[key]) for key in training}
    validation_groups = {passport_key(labels[key]) for key in validation}
    assert training.keys().isdisjoint(validation)
    assert training_groups.isdisjoint(validation_groups)

    model = build_model(training_labels.values(), "outer_train_only")
    processor = Processor(policy, model)
    cached = {key: processor.candidates(row) for key, row in documents.items()}
    _, train_metrics = evaluate_cached(processor, training, training_labels, cached, policy)
    results, validation_metrics = evaluate_cached(processor, validation, labels, cached, policy)
    report = {
        "fold": fold,
        "training": train_metrics,
        "validation": validation_metrics,
        "document_overlap": 0,
        "passport_overlap": 0,
        "model_training_documents": len(training),
    }
    return results, report


def evaluate_repeat(documents, labels, policy, folds, seed):
    assignment = grouped_folds(documents, labels, folds, seed)
    results = {}
    reports = []
    for fold in range(folds):
        predictions, report = evaluate_fold(documents, labels, assignment, fold, policy)
        assert results.keys().isdisjoint(predictions)
        results.update(predictions)
        reports.append(report)
        train = report["training"]["totals"]
        validation = report["validation"]["totals"]
        print(f"seed {seed}, fold {fold + 1}/{folds}: "
              f"train {train['accepted']} accepted, {train['bad']} errors; "
              f"validation {validation['accepted']} accepted, {validation['bad']} errors", flush=True)
    assert results.keys() == documents.keys()
    metrics = summarize(documents, labels, results)
    metrics.update({
        "seed": seed,
        "mode": "grouped_cv",
        "group_risk": grouped_metrics(documents, labels, results),
        "risk_by_source": {
            source: grouped_metrics({key: row for key, row in documents.items()
                                     if row["source"] == source}, labels, results)
            for source in sorted({row["source"] for row in documents.values()})
        },
        "folds": reports,
        "fold_assignment": assignment,
    })
    return results, metrics


def main():
    parser = argparse.ArgumentParser(description="Repeated passport-grouped CV with fixed thresholds")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--mode", choices=("grouped_cv",), default="grouped_cv")
    parser.add_argument("--output", type=Path, default=ROOT / "validation/cv_final")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")

    documents, labels = read_data()
    policy = json.loads((ROOT / "policy.json").read_text(encoding="utf-8"))
    files = ("postprocess.py", "fit_model.py", "cv_tools.py", "run_cv.py", "validate.py",
             "policy.json", "reference/subdivisions.csv", "data/train_predictions.jsonl",
             "data/train_labels.jsonl")
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}
    protocol = {
        "folds": args.folds,
        "repeats": args.repeats,
        "seed": args.seed,
        "mode": "grouped_cv",
        "policy": policy,
        "sha256": hashes,
        "limitations": "Vocabularies exclude outer validation records. Thresholds are fixed. "
                       "Rules were developed using these data, so this is not a blind audit. "
                       "Repeated predictions are not independent observations.",
    }
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists() and not args.overwrite:
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        if previous != protocol:
            parser.error("Existing CV protocol differs; use a new --output directory or --overwrite")
    save_json(protocol_path, protocol)

    runs = []
    accepted_ids = set()
    bad_ids = set()
    for repeat in range(args.repeats):
        results, metrics = evaluate_repeat(documents, labels, policy, args.folds,
                                           args.seed + repeat * 9973)
        metrics["repeat"] = repeat
        directory = args.output / "grouped_cv" / f"repeat-{repeat}"
        save_json(directory / "metrics.json", metrics)
        save_jsonl(directory / "answers.jsonl", [answer for answer, _ in results.values()])
        save_jsonl(directory / "labels.jsonl", labels.values())
        runs.append(metrics)
        accepted_ids.update(key for key, (answer, _) in results.items() if answer["decision"] == "accept")
        bad_ids.update(row["doc_id"] for row in metrics["bad_documents"])

    train_metrics = [fold["training"]["totals"] for run in runs for fold in run["folds"]]
    mode_summary = {
        "validation_per_repeat": [
            {"repeat": run["repeat"], "seed": run["seed"], **run["totals"],
             "by_source": run["by_source"], "group_risk": run["group_risk"],
             "bad_documents": run["bad_documents"]}
            for run in runs
        ],
        "validation_accepted_range": [min(run["totals"]["accepted"] for run in runs),
                                      max(run["totals"]["accepted"] for run in runs)],
        "validation_bad_each_repeat": [run["totals"]["bad"] for run in runs],
        "unique_accepted_any_repeat": len(accepted_ids),
        "unique_bad_any_repeat": len(bad_ids),
        "bad_doc_ids": sorted(bad_ids),
        "train_accepts_per_fold_mean": statistics.mean(row["accepted"] for row in train_metrics),
        "train_bad_fold_evaluations": sum(row["bad"] for row in train_metrics),
    }
    for name, expected in hashes.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("File changed during CV: " + name)
    summary = {
        "protocol": protocol,
        "unique_documents": len(documents),
        "outer_prediction_evaluations": len(documents) * args.repeats,
        "repeated_predictions_are_independent": False,
        "modes": {"grouped_cv": mode_summary},
    }
    save_json(args.output / "summary.json", summary)
    print(json.dumps(mode_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
