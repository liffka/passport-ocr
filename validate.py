import argparse
from collections import Counter, defaultdict
from datetime import date, timedelta
import json
from pathlib import Path

from postprocess import FIELDS, norm, parse_date


def gold_valid(fields):
    birth = parse_date(fields.get("Birth_date"))
    issue = parse_date(fields.get("Issue_date"))
    if birth is None or issue is None:
        return False

    today = date(2026, 10, 1)

    def anniversary(age):
        year = birth.year + age
        leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        if birth.month == 2 and birth.day == 29 and not leap:
            return date(year, 3, 1)
        return date(year, birth.month, birth.day)

    if birth < date(1900, 1, 1) or birth > today:
        return False
    if issue < date(1997, 10, 1) or issue > today or issue < anniversary(14):
        return False
    return all(issue >= anniversary(age) or today <= anniversary(age) + timedelta(days=90)
               for age in (20, 45))


def summarize(predictions, labels, results):
    totals = Counter({key: 0 for key in (
        "accepted", "bad", "good", "field_error_documents", "invalid_accepted"
    )})
    totals["documents"] = len(predictions)
    sources = defaultdict(Counter)
    field_errors = Counter()
    reasons = Counter()
    corrections = Counter()
    bad_documents = []

    for doc_id, document in predictions.items():
        answer, evidence = results[doc_id]
        expected = labels[doc_id]["fields"]
        valid = gold_valid(expected)
        totals["valid_ground_truth"] += valid
        source = sources[document["source"]]
        source["documents"] += 1
        reasons.update(evidence.get("reasons", []))
        if answer["decision"] != "accept":
            continue

        actual = answer.get("fields") or {}
        wrong = [field for field in FIELDS if norm(actual.get(field)) != norm(expected[field])]
        bad = bool(wrong or not valid)
        totals["accepted"] += 1
        totals["bad"] += bad
        totals["good"] += not bad
        totals["field_error_documents"] += bool(wrong)
        totals["invalid_accepted"] += not valid
        source["accepted"] += 1
        source["bad"] += bad
        source["good"] += not bad
        field_errors.update(wrong)
        corrections.update(evidence.get("corrections", []))
        if bad:
            bad_documents.append({"doc_id": doc_id, "wrong_fields": wrong, "invalid": not valid})

    totals["acceptance_rate"] = totals["accepted"] / len(predictions) if predictions else 0.
    return {
        "totals": dict(totals),
        "by_source": {key: dict(value) for key, value in sorted(sources.items())},
        "field_errors": dict(field_errors),
        "rejection_reasons": dict(reasons.most_common()),
        "accepted_corrections": dict(corrections),
        "bad_documents": bad_documents,
    }


def read_rows(path):
    with path.open(encoding="utf-8-sig") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    result = {row["doc_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate doc_id in {path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Check answers against labels and passport dates")
    parser.add_argument("answers", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    answers = read_rows(args.answers)
    labels = read_rows(args.labels)
    if answers.keys() != labels.keys():
        parser.error("Answer and label document IDs differ")
    documents = {key: {"source": "all"} for key in labels}
    results = {key: (answer, {}) for key, answer in answers.items()}
    metrics = summarize(documents, labels, results)
    text = json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8", newline="\n")
    print(text, end="")


if __name__ == "__main__":
    main()
