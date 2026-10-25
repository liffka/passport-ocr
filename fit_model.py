import argparse
from collections import defaultdict
import json
from pathlib import Path

from postprocess import ROOT, norm


def build_model(rows, partition="cv_training"):
    issuers = defaultdict(set)
    first_names = defaultdict(set)
    count = 0
    for row in rows:
        fields = row["fields"]
        issuers[fields["Issue_organisation_code"]].add(norm(fields["Issue_organization_ru"]))
        number = "".join(fields["Licence_number"].split())
        first_names[norm(fields["First_name_ru"])].add(number)
        count += 1
    return {
        "training_partition": partition,
        "training_documents": count,
        "issuers": {key: sorted(values) for key, values in sorted(issuers.items())},
        "first_names": {key: len(values) for key, values in sorted(first_names.items())},
    }


def main():
    parser = argparse.ArgumentParser(description="Fit passport issuer vocabularies on train labels")
    parser.add_argument("--labels", type=Path, default=ROOT / "data/train_labels.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "model.json")
    args = parser.parse_args()
    with args.labels.open(encoding="utf-8-sig") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len({row["doc_id"] for row in rows}) != len(rows):
        parser.error("Training labels contain duplicate document IDs")
    model = build_model(rows, "all_train")
    model.update({
        "evaluation_root": "validation/cv_final",
        "evaluation_mode": "grouped_cv",
        "evaluation_primary_repeat": 0,
    })
    args.output.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8", newline="\n")
    print(f"Training documents: {len(rows)}; issuer codes: {len(model['issuers'])}")


if __name__ == "__main__":
    main()
