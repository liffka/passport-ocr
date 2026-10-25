from collections import Counter, defaultdict
import hashlib
import json
import random

from fit_model import build_model
from postprocess import FIELDS, ROOT, Processor, norm
from validate import gold_valid, summarize

SOURCES = ("web", "mobile", "mobile_v2")


def read_data():
    def load(name):
        with (ROOT / "data" / name).open(encoding="utf-8") as stream:
            rows = list(map(json.loads, stream))
        indexed = {r["doc_id"]: r for r in rows}
        if len(indexed) != len(rows):
            raise ValueError("Duplicate doc_id: " + name)
        return indexed
    documents, labels = load("train_predictions.jsonl"), load("train_labels.jsonl")
    if documents.keys() != labels.keys():
        raise ValueError("Prediction and label IDs differ")
    return documents, labels


def passport_key(row):
    number = "".join(row["fields"]["Licence_number"].split())
    if not number:
        raise ValueError("Missing grouping key")
    return number


def grouped_folds(documents, labels, folds, seed):
    if documents.keys() != labels.keys():
        raise ValueError("Split input IDs differ")
    groups = defaultdict(list)
    for key in sorted(documents):
        groups[passport_key(labels[key])].append(key)
    if not 2 <= folds <= len(groups):
        raise ValueError("Need at least one passport group per fold")
    entries = list(groups.values())
    random.Random(seed).shuffle(entries)
    entries.sort(key=len, reverse=True)
    sources = Counter(d["source"] for d in documents.values())
    target = {s: n / folds for s, n in sources.items()}
    sizes, counts = [0] * folds, [Counter() for _ in range(folds)]
    assignment = {}
    for ids in entries:
        added = Counter(documents[k]["source"] for k in ids)
        def delta(index):
            source_cost = sum(((counts[index][s] + n - target[s]) ** 2
                               - (counts[index][s] - target[s]) ** 2) / target[s]
                              for s, n in added.items())
            total_target = len(documents) / folds
            size_cost = ((sizes[index] + len(ids) - total_target) ** 2
                         - (sizes[index] - total_target) ** 2) / total_target
            return source_cost + size_cost, sizes[index], index
        chosen = min(range(folds), key=delta)
        sizes[chosen] += len(ids)
        counts[chosen].update(added)
        assignment.update({k: chosen for k in ids})
    assert set(assignment) == set(documents)
    assert all(len({assignment[k] for k in ids}) == 1 for ids in groups.values())
    assert all(n > 0 for n in sizes)
    return assignment


def evaluate_cached(processor, documents, labels, cached, policy):
    results = {}
    for key, d in documents.items():
        values, evidence = cached[key]
        answer, reasons = processor.decide(d, values, evidence, policy)
        results[key] = answer, {**evidence, "reasons": reasons}
    return results, summarize(documents, labels, results)


def source_trials(source, base):
    key = "name_mrz_min_" + source
    floor = base.get(key, base["name_mrz_min_v2"] if source == "mobile_v2" else base["name_mrz_min"])
    return sorted(set([floor] + [v for v in (.994, .995, .996, .997, .998, .999, .9995, 1., 1.0001) if v >= floor]))


def choose_policy(documents, labels, base, seed, refit_processor=None, refit_cache=None):

    split = grouped_folds(documents, labels, 4, seed)
    fitting = {k: d for k, d in documents.items() if split[k] != 0}
    calibration = {k: d for k, d in documents.items() if split[k] == 0}
    assert set(fitting).isdisjoint(calibration)
    fitting_labels = {k: labels[k] for k in fitting}
    model = build_model(fitting_labels.values(), "inner_fit")
    processor = Processor(base, model)
    cached = {k: processor.candidates(d) for k, d in documents.items()}
    chosen, logs = dict(base), []
    for source in SOURCES:
        fit_source = {k: d for k, d in fitting.items() if d["source"] == source}
        cal_source = {k: d for k, d in calibration.items() if d["source"] == source}
        candidates = []
        for threshold in source_trials(source, base):
            override = dict(base, **{"name_mrz_min_" + source: threshold})
            fit_results, fit_metrics = evaluate_cached(processor, fit_source, labels, cached, override)
            cal_results, cal_metrics = evaluate_cached(processor, cal_source, labels, cached, override)
            row = {"source": source, "threshold": threshold,
                   "fit": fit_metrics["totals"], "calibration": cal_metrics["totals"]}
            if refit_processor is not None:
                all_source = {k: d for k, d in documents.items() if d["source"] == source}
                _, m = evaluate_cached(refit_processor, all_source, labels, refit_cache, override)
                row["refit_train"] = m["totals"]
            logs.append(row)
            if (not row["fit"]["bad"] and not row["calibration"]["bad"]
                    and not row.get("refit_train", {}).get("bad", 0)):
                candidates.append(row)
        if not candidates:
            raise ValueError("Reject-all fallback should always be feasible")
        selected = max(candidates, key=lambda r: (r["calibration"]["accepted"], r["fit"]["accepted"], r["threshold"]))
        chosen["name_mrz_min_" + source] = selected["threshold"]
    return chosen, {"inner_fit_ids": sorted(fitting), "inner_calibration_ids": sorted(calibration),
                    "selection": "Zero errors in inner fit, inner calibration and refit training consistency; maximize inner calibration acceptance, then fit acceptance, then prefer stricter threshold.",
                    "trials": logs}


def grouped_metrics(documents, labels, results):
    groups = defaultdict(list)
    for key in documents:
        if results[key][0]["decision"] == "accept":
            groups[passport_key(labels[key])].append(key)
    wrong = 0
    for ids in groups.values():
        wrong += any(not gold_valid(labels[k]["fields"]) or any(
            norm(results[k][0]["fields"][field]) != norm(labels[k]["fields"][field])
            for field in FIELDS) for k in ids)
    n = len(groups)
    return {"accepted_passport_groups": n, "bad_passport_groups": wrong,
            "zero_error_one_sided_95_upper": 1 - .05 ** (1/n) if n and wrong == 0 else None,
            "risk_note": "Conditional binomial bound only for zero observed errors; assumes independent exchangeable passport groups. Rules were previously developed on these data."}
