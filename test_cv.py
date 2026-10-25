import copy
from collections import Counter
import unittest
from unittest.mock import patch

from cv_tools import (build_model, choose_policy, grouped_folds, grouped_metrics,
                      passport_key, source_trials)
from postprocess import DEFAULT_POLICY, Processor, encode_lower
from test_postprocess import example


def fixture(n=30):
    documents, labels = {}, {}
    for i in range(n):
        document = example()
        key = f"synthetic-{i:03d}"
        document["doc_id"] = key
        document["source"] = ("web", "mobile", "mobile_v2")[i % 3]
        fields = {k: " ".join(l["text"] for l in v["lines"]) for k, v in document["viz"].items()}
        number = f"01 11 {i:06d}"
        fields["Licence_number"] = number
        line = document["viz"]["Licence_number"]["lines"][0]
        line["text"], line["char_conf"] = number, [.9999] * len(number)
        lower = encode_lower(fields)
        document["mrz"]["lines"][1] = {"text": lower, "char_conf": [.9999] * len(lower)}
        documents[key] = document
        labels[key] = {"doc_id": key, "fields": fields}
    return documents, labels


class CVIntegrity(unittest.TestCase):
    def test_duplicate_passports_never_cross_folds_even_across_sources(self):
        documents, labels = fixture()
        labels["synthetic-001"]["fields"]["Licence_number"] = labels["synthetic-000"]["fields"]["Licence_number"]
        folds = grouped_folds(documents, labels, 5, 42)
        self.assertEqual(folds["synthetic-000"], folds["synthetic-001"])
        self.assertEqual(set(folds), set(documents))

    def test_split_is_deterministic_and_input_order_independent(self):
        documents, labels = fixture(60)
        a = grouped_folds(documents, labels, 5, 42)
        b = grouped_folds(dict(reversed(list(documents.items()))), dict(reversed(list(labels.items()))), 5, 42)
        self.assertEqual(a, b)
        self.assertNotEqual(a, grouped_folds(documents, labels, 5, 43))

    def test_sources_balanced_in_outer_folds(self):
        documents, labels = fixture(150)
        assignment = grouped_folds(documents, labels, 5, 7)
        for source in ("web", "mobile", "mobile_v2"):
            counts = Counter(assignment[k] for k, d in documents.items() if d["source"] == source)
            self.assertEqual(set(counts), set(range(5)))
            self.assertLessEqual(max(counts.values())-min(counts.values()), 1)

    def test_too_many_folds_and_mismatched_ids_rejected(self):
        documents, labels = fixture(3)
        with self.assertRaises(ValueError):
            grouped_folds(documents, labels, 4, 1)
        with self.assertRaises(ValueError):
            grouped_folds(documents, {}, 2, 1)

    def test_model_contains_only_supplied_training_words(self):
        _, labels = fixture(3)
        labels["synthetic-001"]["fields"]["First_name_ru"] = "СЕКРЕТНОЕИМЯ"
        labels["synthetic-001"]["fields"]["Issue_organization_ru"] = "СКРЫТЫЙ ОРГАН"
        model = build_model([labels["synthetic-000"], labels["synthetic-002"]])
        self.assertNotIn("СЕКРЕТНОЕИМЯ", model["first_names"])
        self.assertFalse(any("СКРЫТЫЙ ОРГАН" in v for v in model["issuers"].values()))
        self.assertEqual(model["training_documents"], 2)

    def test_repeated_passport_does_not_inflate_name_frequency(self):
        _, labels = fixture(2)
        labels["synthetic-001"]["fields"]["Licence_number"] = "0111000000"
        model = build_model(labels.values())
        self.assertEqual(model["first_names"]["ИВАН"], 1)

    def test_inner_calibration_records_excluded_from_inner_fit(self):
        documents, labels = fixture(24)
        seen = []
        def tracked(rows, partition):
            rows = list(rows)
            seen.extend(r["doc_id"] for r in rows)
            return build_model(rows, partition)
        with patch("cv_tools.build_model", side_effect=tracked):
            policy, log = choose_policy(documents, labels, dict(DEFAULT_POLICY, allow_repairs=False), 19)
        self.assertEqual(set(seen), set(log["inner_fit_ids"]))
        self.assertTrue(set(seen).isdisjoint(log["inner_calibration_ids"]))
        self.assertEqual(set(seen) | set(log["inner_calibration_ids"]), set(documents))
        self.assertFalse(policy["allow_repairs"])

    def test_threshold_trials_never_relax_existing_floor(self):
        base = dict(DEFAULT_POLICY, name_mrz_min_web=.997, name_mrz_min_mobile_v2=1.)
        self.assertTrue(all(v >= .997 for v in source_trials("web", base)))
        self.assertEqual(source_trials("mobile_v2", base), [1., 1.0001])

    def test_risk_bound_counts_passports_not_duplicate_scans(self):
        documents, labels = fixture(10)
        for i in range(5):
            labels[f"synthetic-{i+5:03d}"]["fields"] = copy.deepcopy(labels[f"synthetic-{i:03d}"]["fields"])
        results = {k: ({"doc_id": k, "decision": "accept", "fields": row["fields"]}, {"reasons": []}) for k, row in labels.items()}
        risk = grouped_metrics(documents, labels, results)
        self.assertEqual(risk["accepted_passport_groups"], 5)
        self.assertEqual(risk["bad_passport_groups"], 0)
        self.assertAlmostEqual(risk["zero_error_one_sided_95_upper"], 1-.05**.2)

    def test_literal_mode_rejects_recovered_value(self):
        row = example()
        line = row["viz"]["Licence_number"]["lines"][0]
        line["text"] = line["text"][:-1] + "7"
        line["char_conf"][-1] = .5
        model = {"first_names": {"ИВАН": 1}}
        self.assertEqual(Processor(model=model).process(row)[0]["decision"], "accept")
        answer, evidence = Processor(policy={"allow_repairs": False}, model=model).process(row)
        self.assertEqual(answer["decision"], "reject")
        self.assertIn("repair_disabled_for_safety", evidence["reasons"])


if __name__ == "__main__":
    unittest.main()
