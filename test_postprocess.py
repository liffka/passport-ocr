import copy
from datetime import date, timedelta
import unittest

from postprocess import (AS_OF, FIELDS, NAMES, TO_MRZ, Processor, birthday, checksum,
                         decode_lower, encode_lower, parse_date, six_digit_date,
                         validity_reason, visual)
from validate import gold_valid


def example():
    fields = dict(zip(FIELDS, (
        "ИВАНОВ", "ИВАН", "ИВАНОВИЧ", "МУЖ.", "01.01.1990", "Г. МОСКВА",
        "01.02.2011", "010-001", "01 11 123456", "МВД ПО РЕСП. АДЫГЕЯ")))
    visual_fields = {}
    for i, (k, text) in enumerate(fields.items()):
        y = i * 40
        visual_fields[k] = {"box": [0, y, 500, 30], "conf": .9999,
                            "lines": [{"text": text, "box": [0, y+1, 450, 28],
                                       "char_conf": [.9999] * len(text)}]}
    upper = ("PNRUS" + fields[NAMES[0]].translate(TO_MRZ) + "<<"
             + fields[NAMES[1]].translate(TO_MRZ) + "<"
             + fields[NAMES[2]].translate(TO_MRZ)).ljust(44, "<")
    lower = encode_lower(fields)
    return {"doc_id": "synthetic", "source": "web", "quality": {"sharpness": 200, "doc_px": 1800, "glare": .01},
            "viz": visual_fields, "mrz": {"lines": [
                {"text": s, "char_conf": [.9999] * len(s)} for s in (upper, lower)]}}


class DatesAndValidity(unittest.TestCase):
    def valid_fields(self, birth, issued):
        return {"Birth_date": birth, "Issue_date": issued}

    def test_strict_calendar(self):
        self.assertIsNone(parse_date("31.04.2000"))
        self.assertIsNone(parse_date("29.02.2001"))
        self.assertIsNone(parse_date("1.01.2000"))
        self.assertEqual(parse_date("29.02.2000"), date(2000, 2, 29))

    def test_90_day_boundary_both_ages(self):
        for age in (20, 45):
            birth = date(2000 if age == 20 else 1975, 1, 1)
            milestone = birthday(birth, age)
            issue_age = 14 if age == 20 else 30
            fields = self.valid_fields(birth.strftime("%d.%m.%Y"), birthday(birth, issue_age).strftime("%d.%m.%Y"))
            self.assertIsNone(validity_reason(fields, milestone + timedelta(days=90)))
            self.assertEqual(validity_reason(fields, milestone + timedelta(days=91)), "expired_at_age_" + str(age))

    def test_new_passport_on_birthday(self):
        self.assertIsNone(validity_reason(self.valid_fields("01.01.1980", "01.01.2025")))

    def test_future_and_underage(self):
        self.assertEqual(validity_reason(self.valid_fields("01.01.1990", "02.10.2026")), "future_date")
        self.assertEqual(validity_reason(self.valid_fields("01.01.2010", "31.12.2023")), "issued_before_age_14")

    def test_issue_age_14_boundary(self):
        self.assertIsNone(validity_reason(self.valid_fields("01.01.2010", "01.01.2024")))

    def test_leap_birthday(self):
        self.assertEqual(birthday(date(1980, 2, 29), 45), date(2025, 3, 1))

    def test_century_ambiguity(self):
        self.assertIsNone(six_digit_date("050101", "Birth_date", ""))
        self.assertEqual(six_digit_date("050101", "Birth_date", "01.01.2005"), "01.01.2005")
        self.assertEqual(six_digit_date("990101", "Issue_date", ""), "01.01.1999")

    def test_ocr_corrupted_early_year_does_not_crash(self):
        self.assertEqual(six_digit_date("880101", "Birth_date", "01.01.1088"), "01.01.1988")

    def test_independent_evaluator_agrees_on_date_grid(self):
        for year in (1960, 1975, 1981, 1982, 1990, 2005, 2006, 2007, 2010, 2012):
            for month, day in ((1, 1), (7, 3), (10, 1), (10, 2), (12, 31)):
                for issued_year in (1997, 2000, 2010, 2020, 2025, 2026, 2027):
                    f = self.valid_fields(f"{day:02d}.{month:02d}.{year}", f"01.10.{issued_year}")
                    self.assertEqual(validity_reason(f) is None, gold_valid(f), f)


class MRZAndDecisions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = Processor(model={"first_names": {"ИВАН": 1}})

    def test_known_checksum(self):
        self.assertEqual(checksum("612283968"), "7")
        self.assertEqual(checksum("100610"), "2")

    def test_serial_fourth_digit(self):
        row = example()
        decoded = decode_lower(row["mrz"]["lines"][1])
        self.assertEqual(decoded["Licence_number"], "01 11 123456")
        self.assertEqual(decoded["Issue_organisation_code"], "010-001")

    def test_numeric_letter_checksum_collision_rejected(self):

        row = example()
        lower = row["mrz"]["lines"][1]
        lower["text"] = lower["text"].replace("6", "G", 1)
        self.assertIsNone(decode_lower(lower))

    def test_wrong_composite_rejected(self):
        row = example()
        line = row["mrz"]["lines"][1]
        line["text"] = line["text"][:-1] + str((int(line["text"][-1]) + 1) % 10)
        self.assertIsNone(decode_lower(line))

    def test_clean_document_accepts(self):
        answer, _ = self.processor.process(example())
        self.assertEqual(answer["decision"], "accept")
        self.assertEqual(set(answer["fields"]), set(FIELDS))

    def test_confident_number_conflict_rejects(self):
        row = example()
        row["viz"]["Licence_number"]["lines"][0]["text"] = "01 11 123457"
        answer, _ = self.processor.process(row)
        self.assertEqual(answer["decision"], "reject")
        self.assertNotIn("fields", answer)

    def test_uncertain_number_repaired(self):
        row = example()
        line = row["viz"]["Licence_number"]["lines"][0]
        line["text"] = "01 11 123457"
        line["char_conf"][-1] = .5
        answer, evidence = self.processor.process(row)
        self.assertEqual(answer["decision"], "accept")
        self.assertEqual(answer["fields"]["Licence_number"], "01 11 123456")
        self.assertIn("Licence_number:mrz", evidence["corrections"])

    def test_missing_name_recovered(self):
        row = example()
        row["viz"]["First_name_ru"] = None
        answer, _ = self.processor.process(row)
        self.assertEqual(answer["decision"], "accept")
        self.assertEqual(answer["fields"]["First_name_ru"], "ИВАН")

    def test_confident_name_conflict_rejects(self):
        row = example()
        row["viz"]["First_name_ru"]["lines"][0]["text"] = "ИГАН"
        self.assertEqual(self.processor.process(row)[0]["decision"], "reject")

    def test_missing_mrz_rejects(self):
        row = example()
        row["mrz"] = None
        self.assertEqual(self.processor.process(row)[0]["decision"], "reject")

    def test_misaligned_confidence_rejects(self):
        row = example()
        row["mrz"]["lines"][0]["char_conf"].pop()
        self.assertEqual(self.processor.process(row)[0]["decision"], "reject")

    def test_low_confidence_mrz_name_character_repaired(self):
        row = example()
        upper = row["mrz"]["lines"][0]
        upper["text"] = upper["text"][:5] + "G" + upper["text"][6:]
        upper["char_conf"][5] = .5
        answer, evidence = self.processor.process(row)
        self.assertEqual(answer["decision"], "accept")
        self.assertIn("mrz_upper:visual_repair", evidence["corrections"])

    def test_missing_top_line_detected_despite_confidence_one(self):
        row = example()
        field = row["viz"]["Birth_place_ru"]
        field["box"][1] -= 30
        field["box"][3] += 30
        field["lines"][0]["char_conf"] = [1.] * len(field["lines"][0]["text"])
        self.assertFalse(visual(field)[2])
        self.assertEqual(self.processor.process(row)[0]["decision"], "reject")

    def test_missing_internal_line_detected(self):
        row = example()
        field = row["viz"]["Birth_place_ru"]
        field["box"][3] = 90
        other = copy.deepcopy(field["lines"][0])
        other["box"][1] += 60
        field["lines"].append(other)
        self.assertFalse(visual(field)[2])

    def test_issuer_punctuation_not_canonicalized(self):
        row = example()
        text = "МВД ПО РЕСП. АДЫГЕЯ."
        field = row["viz"]["Issue_organization_ru"]
        field["lines"][0]["text"] = text
        field["lines"][0]["char_conf"] = [.9999] * len(text)
        answer, _ = self.processor.process(row)
        self.assertEqual(answer["decision"], "accept")
        self.assertEqual(answer["fields"]["Issue_organization_ru"], text)

    def test_v2_stricter_free_text_threshold(self):
        row = example()
        row["viz"]["Birth_place_ru"]["lines"][0]["char_conf"][0] = .97
        self.assertEqual(self.processor.process(row)[0]["decision"], "accept")
        row["source"] = "mobile_v2"
        self.assertEqual(self.processor.process(row)[0]["decision"], "reject")

    def test_issuer_valid_grammatical_form_not_replaced(self):
        row = example()
        text = "ОТДЕЛ УФМС РОССИИ ПО Г. МОСКВЕ"
        field = row["viz"]["Issue_organization_ru"]
        field["lines"][0]["text"] = text
        field["lines"][0]["char_conf"] = [.9999] * len(text)
        processor = Processor(model={"issuers": {"010-001": ["ОТДЕЛОМ УФМС РОССИИ ПО Г. МОСКВЕ"]}})
        values, evidence = processor.candidates(row)
        self.assertEqual(values["Issue_organization_ru"], text)
        self.assertNotIn("Issue_organization_ru:dictionary", evidence["corrections"])

    def test_v2_uses_both_visual_and_mrz_confidence(self):
        row = example()
        row["source"] = "mobile_v2"
        for line in row["mrz"]["lines"]:
            line["char_conf"] = [1.] * len(line["text"])
        processor = Processor(policy={"name_mrz_min_mobile_v2": 1.}, model={"first_names": {"ИВАН": 1}})
        self.assertEqual(processor.process(row)[0]["decision"], "reject")
        for key in NAMES:
            row["viz"][key]["lines"][0]["char_conf"] = [1.] * len(row["viz"][key]["lines"][0]["text"])
        self.assertEqual(processor.process(row)[0]["decision"], "accept")

    def test_unknown_first_name_is_rejected_without_autocorrection(self):
        row = example()
        processor = Processor(model={"first_names": {"ПЕТР": 1}})
        answer, evidence = processor.process(row)
        self.assertEqual(answer["decision"], "reject")
        self.assertIn("unseen_first_name", evidence["reasons"])

    def test_spelling_fix_must_not_add_grammatical_ending(self):
        row = example()
        text = "ОДЕЛЕНИЕ УФМС РОССИИ ПО Г. МОСКВЕ"
        field = row["viz"]["Issue_organization_ru"]
        field["lines"][0]["text"] = text
        field["lines"][0]["char_conf"] = [.9999] * len(text)
        processor = Processor(model={"issuers": {"010-001": ["ОТДЕЛЕНИЕМ УФМС РОССИИ ПО Г. МОСКВЕ"]}})
        values, evidence = processor.candidates(row)
        self.assertEqual(values["Issue_organization_ru"], text)
        self.assertNotIn("Issue_organization_ru:dictionary", evidence["corrections"])


if __name__ == "__main__":
    unittest.main()
