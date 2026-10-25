from collections import Counter
import json
import math

from postprocess import ROOT


def predictive(n, errors, checked):

    a, b = errors + .5, checked - errors + .5
    def log_beta(x, y):
        return math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y)
    return [math.exp(math.lgamma(n+1) - math.lgamma(k+1) - math.lgamma(n-k+1)
                     + log_beta(a+k, b+n-k) - log_beta(a, b)) for k in range(n+1)]


def main():
    with (ROOT / "data/test_predictions.jsonl").open(encoding="utf-8") as f:
        documents = {r["doc_id"]: r for r in map(json.loads, f)}
    with (ROOT / "answers.jsonl").open(encoding="utf-8") as f:
        answers = {r["doc_id"]: r for r in map(json.loads, f)}
    assert documents.keys() == answers.keys()
    counts = Counter(documents[k]["source"] for k, a in answers.items() if a["decision"] == "accept")
    model = json.loads((ROOT / "model.json").read_text(encoding="utf-8"))
    repeat = model.get("evaluation_primary_repeat", 0)
    cv_path = ((ROOT / model["evaluation_root"] / model["evaluation_mode"] / f"repeat-{repeat}" / "metrics.json")
               if "evaluation_root" in model else ROOT / "validation/cross_validation.json")
    cv = json.loads(cv_path.read_text(encoding="utf-8"))
    distribution = [1.]
    by_source = {}
    expected = 0
    for source in sorted(counts):
        risk = cv.get("risk_by_source", {}).get(source)
        checked = risk["accepted_passport_groups"] if risk else cv["by_source"][source]["accepted"]
        errors = risk["bad_passport_groups"] if risk else cv["by_source"][source].get("bad", 0)
        n = counts[source]
        mean = (errors + .5) / (checked + 1)
        expected += n * mean
        by_source[source] = {"test_accepted": n, "cv_accepted": checked, "cv_errors": errors,
                             "smoothed_error_rate": mean, "expected_test_errors": n * mean}
        probabilities = predictive(n, errors, checked)
        convolved = [0.] * (len(distribution) + n)
        for i, p in enumerate(distribution):
            for j, q in enumerate(probabilities):
                convolved[i+j] += p*q
        distribution = convolved
    cumulative, upper = 0., len(distribution)-1
    for k, probability in enumerate(distribution):
        cumulative += probability
        if cumulative >= .95:
            upper = k
            break
    result = {"test_documents": len(documents), "test_accepted": sum(counts.values()),
              "test_rejected": len(documents)-sum(counts.values()),
              "expected_errors": expected, "rounded_error_forecast": math.floor(expected+.5),
              "probability_zero_errors": distribution[0],
              "model_based_95_percent_predictive_interval": [0, upper],
              "by_source": by_source,
              "cv_file": str(cv_path.relative_to(ROOT)),
              "primary_repeat": repeat,
              "repeats_pooled_as_independent": False,
              "method": "Source-level Beta-binomial predictions with Jeffreys Beta(1/2,1/2) priors from predeclared primary repeat 0 of outer CV for the deployed grouped CV procedure. Accepted passport groups are counted once per source; repeated predictions are not pooled.",
              "limitations": "Rules were revised after the initial holdout. Cross-validation is post-development; the interval is conditional on exchangeability, source stability, independence and the chosen prior. It is not a validated confidence guarantee. No test labels exist locally."}
    (ROOT / "validation/forecast.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
