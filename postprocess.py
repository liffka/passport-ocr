from __future__ import annotations

import argparse
from calendar import isleap, monthrange
from collections import defaultdict
import csv
from datetime import date, timedelta
import difflib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
AS_OF = date(2026, 10, 1)
FIELDS = ("Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru",
          "Birth_date", "Birth_place_ru", "Issue_date", "Issue_organisation_code",
          "Licence_number", "Issue_organization_ru")
NAMES = FIELDS[:3]
RU = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
MRZ_ALPHABET = "ABVGDE2JZIQKLMNOPRSTUFHC34WXY9678"
TO_MRZ = str.maketrans(RU, MRZ_ALPHABET)
FROM_MRZ = str.maketrans(MRZ_ALPHABET, RU)
HOMOGLYPHS = str.maketrans("ABCEHKMOPTXY", "АВСЕНКМОРТХУ")
DEFAULT_POLICY = {
    "birth_min": .95, "birth_min_v2": .998,
    "issuer_min": .95, "issuer_min_v2": .998,
    "name_mrz_min": .995, "name_mrz_min_v2": .998,
    "name_visual_min": .99,
    "require_known_first_name": True,
    "numeric_mrz_min": .90, "repair_mrz_min": .995,
    "issuer_max_edits": 2, "issuer_margin": 2,
}


def norm(value):
    return " ".join(str(value or "").split()).upper()


def parse_date(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{2}\.[0-9]{2}\.[0-9]{4}", value):
        return None
    day, month, year = map(int, value.split("."))
    if not 1 <= year <= 9999 or not 1 <= month <= 12:
        return None
    if not 1 <= day <= monthrange(year, month)[1]:
        return None
    return date(year, month, day)


def birthday(birth, age):
    year = birth.year + age
    if birth.month == 2 and birth.day == 29 and not isleap(year):
        return date(year, 3, 1)
    return date(year, birth.month, birth.day)


def date6(value):

    return f"{value.year % 100:02d}{value.month:02d}{value.day:02d}"


def validity_reason(fields, as_of=AS_OF):
    birth = parse_date(fields.get("Birth_date"))
    issued = parse_date(fields.get("Issue_date"))
    if birth is None or issued is None:
        return "invalid_calendar_date"
    if birth < date(1900, 1, 1) or issued < date(1997, 10, 1):
        return "unsupported_date_range"
    if issued > as_of or birth > as_of:
        return "future_date"
    if issued < birthday(birth, 14):
        return "issued_before_age_14"
    for age in (20, 45):
        milestone = birthday(birth, age)
        if issued < milestone and as_of > milestone + timedelta(days=90):
            return "expired_at_age_" + str(age)
    return None


def checksum(text):
    if not re.fullmatch(r"[A-Z0-9<]*", text):
        return "?"
    values = [int(c) if c.isdigit() else 0 if c == "<" else ord(c) - 55 for c in text]
    return str(sum(v * (7, 3, 1)[i % 3] for i, v in enumerate(values)) % 10)


def confidence(line):
    text = line.get("text", "")
    values = line.get("char_conf", [])
    if len(values) != len(text) or not values:
        return 0.0
    if any(not isinstance(c, (float, int)) or not 0 <= c <= 1 for c in values):
        return 0.0
    return min(values)


def visual(field):
    if not field or not field.get("lines"):
        return "", 0.0, False
    lines = sorted(field["lines"], key=lambda l: (l["box"][1], l["box"][0]))
    text = norm(" ".join(l["text"] for l in lines))
    minimum = min(confidence(l) for l in lines)
    box = field.get("box", [])
    if len(box) != 4 or any(len(l.get("box", [])) != 4 for l in lines):
        return text, minimum, False
    height = sum(l["box"][3] for l in lines) / len(lines)
    if height <= 0:
        return text, minimum, False
    top = lines[0]["box"][1] - box[1]
    bottom = box[1] + box[3] - lines[-1]["box"][1] - lines[-1]["box"][3]
    complete = (top < .5 * height and bottom < .5 * height
                and all(b["box"][1] - a["box"][1] < 1.6 * height
                        for a, b in zip(lines, lines[1:])))
    return text, minimum, complete


def visual_character_scores(field):

    if not field or not field.get("lines"):
        return []
    chars = []
    for line in sorted(field["lines"], key=lambda l: (l["box"][1], l["box"][0])):
        if len(line["text"]) != len(line.get("char_conf", [])):
            return []
        if chars:
            chars.append((" ", 1.0))
        chars.extend(zip(line["text"].upper(), line["char_conf"]))
    result, pending = [], []
    for char, score in chars:
        if char.isspace():
            pending.append(score)
        else:
            if result and pending:
                result.append(min(pending))
            pending.clear()
            result.append(score)
    return result


def mrz_lines(document):
    lines = (document.get("mrz") or {}).get("lines", [])
    top = next((l for l in lines if l.get("text", "").startswith("PNRUS")), None)
    bottom = next((l for l in lines if len(l.get("text", "")) == 44
                   and l["text"][10:13] == "RUS"), None)
    return top, bottom


def decode_lower(line):

    if line is None or confidence(line) == 0:
        return None
    text = line["text"]
    if not re.fullmatch(r"[0-9]{10}RUS[0-9]{7}[MF]<{7}[0-9]{13}<[0-9]{2}", text):
        return None
    if not all((checksum(text[:9]) == text[9],
                checksum(text[13:19]) == text[19],
                checksum(text[28:42]) == text[42],
                checksum(text[:10] + text[13:20] + text[21:43]) == text[43])):
        return None
    digits = text[:3] + text[28] + text[3:9]
    return {
        "Licence_number": digits[:2] + " " + digits[2:4] + " " + digits[4:],
        "Birth_date": text[13:19], "Issue_date": text[29:35],
        "Issue_organisation_code": text[35:38] + "-" + text[38:41],
        "Sex_ru": {"M": "МУЖ.", "F": "ЖЕН."}[text[20]],
    }


def encode_lower(fields):
    number = fields["Licence_number"].replace(" ", "")
    code = fields["Issue_organisation_code"].replace("-", "")
    birth, issue = parse_date(fields["Birth_date"]), parse_date(fields["Issue_date"])
    if not (re.fullmatch(r"[0-9]{10}", number) and re.fullmatch(r"[0-9]{6}", code)
            and birth and issue and fields["Sex_ru"] in ("МУЖ.", "ЖЕН.")):
        return None
    serial = number[:3] + number[4:]
    born = date6(birth)
    optional = number[3] + date6(issue) + code + "<"
    result = (serial + checksum(serial) + "RUS" + born + checksum(born)
              + ("M" if fields["Sex_ru"] == "МУЖ." else "F") + "<<<<<<<"
              + optional + checksum(optional))
    return result + checksum(result[:10] + result[13:20] + result[21:43])


def repair_lower(line, fields, observed):

    if line is None or confidence(line) == 0:
        return None
    expected = encode_lower(fields)
    if expected is None:
        return None
    keys = ("Licence_number", "Birth_date", "Issue_date", "Sex_ru", "Issue_organisation_code")
    if any(observed[k][1] < .98 or not observed[k][2] for k in keys):
        return None
    text = line["text"]
    if edit_distance(text, expected) > 2:
        return None
    cs = line["char_conf"]
    repaired_cs = []
    for tag, a, b, c, d in difflib.SequenceMatcher(None, text, expected, autojunk=False).get_opcodes():
        if tag == "equal":
            repaired_cs.extend(cs[a:b])
        else:

            changed = cs[a:b] or cs[max(0, a-1):min(len(cs), a+1)]
            if not changed or max(changed) >= .95:
                return None
            repaired_cs.extend([min(observed[k][1] for k in keys)] * (d-c))
    repaired = {"text": expected, "char_conf": repaired_cs}
    return repaired if decode_lower(repaired) else None


def decode_names(line):
    if not line or confidence(line) == 0 or len(line["text"]) != 44:
        return None
    text = line["text"]
    if not text.startswith("PNRUS") or not re.fullmatch(r"[A-Z0-9<]{44}", text):
        return None
    if "<<" not in text[5:]:
        return None
    last, rest = text[5:].split("<<", 1)
    tokens = rest.rstrip("<").split("<")
    if not last or not tokens[0] or len(tokens) > 2:
        return None
    first, middle = tokens[0], tokens[1] if len(tokens) == 2 else ""
    decoded = [s.translate(FROM_MRZ) for s in (last, first, middle)]
    if any(s and not re.fullmatch(r"[А-ЯЁ]+", s) for s in decoded):
        return None

    return dict(zip(NAMES, decoded)) if text.endswith("<") else None


def repair_upper(line, expected, observed):
    if not line or confidence(line) == 0 or edit_distance(line["text"], expected) > 2:
        return None
    present = [observed[k] for k in NAMES if observed[k][0]]
    if any(v[1] < .995 or not v[2] for v in present):
        return None
    score = min((v[1] for v in present), default=0)
    scores = line["char_conf"]
    repaired = []
    for tag, a, b, c, d in difflib.SequenceMatcher(None, line["text"], expected, autojunk=False).get_opcodes():
        if tag == "equal":
            repaired.extend(scores[a:b])
        else:
            changed = scores[a:b] or scores[max(0, a-1):min(len(scores), a+1)]
            if not changed or max(changed) >= .95:
                return None
            repaired.extend([score] * (d-c))
    return {"text": expected, "char_conf": repaired}


def six_digit_date(text, kind, visual_value):
    visual_date = parse_date(visual_value)
    if visual_date and 1900 <= visual_date.year <= AS_OF.year and date6(visual_date) == text:
        return visual_value
    low, high = (1900, AS_OF.year - 14) if kind == "Birth_date" else (1997, AS_OF.year)
    candidates = []
    for century in (1900, 2000):
        year = century + int(text[:2])
        value = parse_date(f"{text[4:6]}.{text[2:4]}.{year}")
        if value is None:
            continue
        if low <= year <= high and value <= AS_OF:
            candidates.append(value.strftime("%d.%m.%Y"))
    return candidates[0] if len(candidates) == 1 else None


def format_visual(key, value):
    if key in NAMES:
        return value.translate(HOMOGLYPHS)
    digits = re.sub(r"[^0-9]", "", value)
    if key == "Licence_number" and len(digits) == 10:
        return digits[:2] + " " + digits[2:4] + " " + digits[4:]
    if key == "Issue_organisation_code" and len(digits) == 6:
        return digits[:3] + "-" + digits[3:]
    if key in ("Birth_date", "Issue_date") and len(digits) == 8:
        return digits[:2] + "." + digits[2:4] + "." + digits[4:]
    return value


def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


class Processor:
    def __init__(self, policy=None, model=None):
        self.policy = dict(DEFAULT_POLICY, **(policy or {}))
        self.issuers = defaultdict(set)
        with (ROOT / "reference" / "subdivisions.csv").open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                self.issuers[row["code"]].add(norm(row["name"]))
        for code, names in (model or {}).get("issuers", {}).items():
            self.issuers[code].update(names)
        self.issuer_words = {word for names in self.issuers.values() for name in names
                             for word in re.findall(r"[А-ЯЁ]+", name)}
        self.first_names = set((model or {}).get("first_names", {}))
        self.word_neighbours = {}

    def known_one_edit(self, word):
        if word in self.word_neighbours:
            return self.word_neighbours[word]
        choices = {word[:i] + word[i+1:] for i in range(len(word))}
        choices.update(word[:i] + char + word[i+1:]
                       for i in range(len(word)) for char in RU if char != word[i])
        choices.update(word[:i] + char + word[i:]
                       for i in range(len(word)+1) for char in RU)
        neighbours = choices & self.issuer_words
        self.word_neighbours[word] = neighbours
        return neighbours

    def _resolve_issuer(self, document, values, observed, evidence):

        code = values["Issue_organisation_code"]
        text = values["Issue_organization_ru"]
        candidates = self.issuers.get(code, set())
        evidence["issuer_known"] = text in candidates
        if candidates and text not in candidates and observed["Issue_organization_ru"][2]:
            shortlist = [s for s in candidates if abs(len(s) - len(text)) <= 2]
            ranked = sorted((edit_distance(text, s), s) for s in shortlist)
            if ranked:
                best, candidate = ranked[0]
                margin = ranked[1][0] - best if len(ranked) > 1 else 99
                evidence["issuer_distance"] = best
                evidence["issuer_margin"] = margin
                same_punctuation = (re.sub(r"[А-ЯЁ]", "", text)
                                    == re.sub(r"[А-ЯЁ]", "", candidate))
                original_words = re.findall(r"[А-ЯЁ]+", text)
                proposed_words = re.findall(r"[А-ЯЁ]+", candidate)
                changes_known_word = (len(original_words) != len(proposed_words)
                                      or any(a != b and a in self.issuer_words
                                             for a, b in zip(original_words, proposed_words)))

                inflections = {"ИЙ", "ИМ", "ЫЙ", "ЫМ", "ОЙ", "ОМ", "ЕМ", "ОЕ", "АЯ", "ОЮ", "УЮ"}
                grammatical_variant = any(a != b and len(a) > 4 and len(b) > 4
                                          and a[:-2] == b[:-2]
                                          and a[-2:] in inflections and b[-2:] in inflections
                                          for a, b in zip(original_words, proposed_words))
                competing_spelling = False
                if same_punctuation and not changes_known_word and not grammatical_variant and best <= 2:
                    for original, proposed in zip(original_words, proposed_words):
                        if original == proposed:
                            continue
                        neighbours = self.known_one_edit(original)
                        distance = edit_distance(original, proposed)
                        if ((distance > 1 and neighbours)
                                or (distance == 1 and neighbours - {proposed})):
                            competing_spelling = True
                if (best <= self.policy["issuer_max_edits"] and margin >= self.policy["issuer_margin"]
                        and same_punctuation and not changes_known_word and not grammatical_variant
                        and not competing_spelling):
                    values["Issue_organization_ru"] = candidate
                    evidence["issuer_known"] = True
                    evidence["corrections"].append("Issue_organization_ru:dictionary")
                    scores = visual_character_scores(document["viz"]["Issue_organization_ru"])
                    if len(scores) == len(text):
                        unchanged = [score for tag, a, b, _, _ in
                                     difflib.SequenceMatcher(None, text, candidate, autojunk=False).get_opcodes()
                                     if tag == "equal" for score in scores[a:b]]
                        evidence["issuer_residual_min"] = min(unchanged, default=0)

    def candidates(self, document):

        evidence = {"reasons": [], "corrections": []}
        values = {}
        observed = {}
        for key in FIELDS:
            raw, minimum, complete = visual(document.get("viz", {}).get(key))
            value = format_visual(key, raw)
            values[key] = value
            observed[key] = (raw, minimum, complete)
            evidence[key] = {"min": minimum, "complete": complete}
            if value != raw:
                evidence["corrections"].append(key + ":format")

        upper, lower = mrz_lines(document)
        decoded = decode_lower(lower)
        if decoded is None:

            possible = (document.get("mrz") or {}).get("lines", [])
            damaged = lower or next((l for l in possible if "RUS" in l.get("text", "")
                                    and not l["text"].startswith("PNRUS")), None)
            fixed = repair_lower(damaged, values, observed)
            if fixed:
                lower, decoded = fixed, decode_lower(fixed)
                evidence["corrections"].append("mrz_lower:visual_repair")
        evidence["lower_min"] = confidence(lower) if decoded else 0
        if decoded is None:
            evidence["reasons"].append("mrz_lower_unverified")
        else:
            for key, mrz_value in decoded.items():
                if key in ("Birth_date", "Issue_date"):
                    mrz_value = six_digit_date(mrz_value, key, values[key])
                if mrz_value is None:
                    evidence["reasons"].append(key + ":ambiguous_century")
                    continue
                if values[key] != mrz_value:


                    if observed[key][1] < .95:
                        values[key] = mrz_value
                        evidence["corrections"].append(key + ":mrz")
                        evidence[key]["repaired"] = True
                        slices = {"Licence_number": (0, 10), "Birth_date": (13, 20),
                                  "Issue_date": (28, 43), "Issue_organisation_code": (28, 43),
                                  "Sex_ru": (20, 21)}
                        start, end = slices[key]
                        scores = lower["char_conf"][start:end]
                        if key == "Licence_number":
                            scores = scores + lower["char_conf"][28:43]
                        evidence[key]["repair_min"] = min(scores)
                    else:
                        evidence["reasons"].append(key + ":conflict")

        decoded_names = decode_names(upper)
        if decoded_names and confidence(upper) >= .995:
            for key, mrz_value in decoded_names.items():
                if (values[key] != mrz_value and observed[key][1] < .95
                        and (not values[key] or edit_distance(values[key], mrz_value) <= 2)):
                    values[key] = mrz_value
                    evidence["corrections"].append(key + ":mrz")
                    evidence[key]["name_repaired"] = True
        expected = ("PNRUS" + values[NAMES[0]].translate(TO_MRZ) + "<<"
                    + values[NAMES[1]].translate(TO_MRZ) + "<"
                    + values[NAMES[2]].translate(TO_MRZ))
        expected = expected.replace(" ", "<").replace("-", "<").ljust(44, "<")[:44]
        if upper is None or upper["text"] != expected:
            possible = (document.get("mrz") or {}).get("lines", [])
            damaged = upper or next((l for l in possible if l.get("text", "").startswith("PN")), None)
            fixed = repair_upper(damaged, expected, observed)
            if fixed:
                upper = fixed
                evidence["corrections"].append("mrz_upper:visual_repair")
        if upper and confidence(upper) > 0:
            name_chars_conf = [c for ch, c in zip(upper["text"], upper["char_conf"]) if ch != "<"]
            evidence["upper_min"] = min(name_chars_conf, default=confidence(upper))
        else:
            evidence["upper_min"] = confidence(upper) if upper else 0
        evidence["name_visual_min"] = min([
            observed[k][1] for k in NAMES
            if values[k] and not evidence[k].get("name_repaired")], default=1.)
        evidence["name_min"] = min(evidence["upper_min"], evidence["name_visual_min"])
        evidence["names_agree"] = bool(upper and upper["text"] == expected)
        if not evidence["names_agree"]:
            evidence["reasons"].append("names_mrz_conflict")
        for key in NAMES:
            if key != "Middle_name_ru" and not values[key]:
                evidence["reasons"].append(key + ":missing")
            if values[key] and not re.fullmatch(r"[А-ЯЁ]+(?:[ -][А-ЯЁ]+)*", values[key]):
                evidence["reasons"].append(key + ":format")
        position = 5
        for index, key in enumerate(NAMES):
            position += len(values[key])
            if position > 44 and (observed[key][1] < .995 or not observed[key][2]):
                evidence["reasons"].append(key + ":truncated_in_mrz")
            position += 2 if index == 0 else 1

        for key in ("Birth_place_ru", "Issue_organization_ru"):
            if not observed[key][2] or not values[key]:
                evidence["reasons"].append(key + ":incomplete")

        self._resolve_issuer(document, values, observed, evidence)

        reason = validity_reason(values)
        if reason:
            evidence["reasons"].append(reason)
        return values, evidence

    def decide(self, document, values, evidence, policy=None):
        p = self.policy if policy is None else dict(self.policy, **policy)
        reasons = list(evidence["reasons"])
        suffix = "_v2" if document.get("source") == "mobile_v2" else ""
        if document.get("source") not in ("web", "mobile", "mobile_v2"):
            reasons.append("unknown_source")
        if not p.get("allow_repairs", True) and any(
                not event.endswith(":format") for event in evidence["corrections"]):
            reasons.append("repair_disabled_for_safety")
        if p["require_known_first_name"] and values["First_name_ru"] not in self.first_names:
            reasons.append("unseen_first_name")
        name_threshold = p.get("name_mrz_min_" + document.get("source", ""), p["name_mrz_min" + suffix])
        name_min = evidence["name_min"] if suffix else evidence["upper_min"]
        if name_min < name_threshold:
            reasons.append("names_mrz_confidence")
        if not suffix and evidence["name_visual_min"] < p["name_visual_min"]:
            reasons.append("names_visual_confidence")
        if evidence["lower_min"] < p["numeric_mrz_min"]:
            reasons.append("numbers_mrz_confidence")
        if any(evidence[key].get("repaired") and evidence[key]["repair_min"] < p["repair_mrz_min"]
               for key in FIELDS):
            reasons.append("repair_mrz_confidence")
        if evidence["Birth_place_ru"]["min"] < p["birth_min" + suffix]:
            reasons.append("birthplace_confidence")
        issuer_threshold = .90 if evidence["issuer_known"] else p["issuer_min" + suffix]
        issuer_min = evidence.get("issuer_residual_min", evidence["Issue_organization_ru"]["min"])
        if issuer_min < issuer_threshold:
            reasons.append("issuer_confidence")
        answer = {"doc_id": document["doc_id"], "decision": "reject" if reasons else "accept"}
        if not reasons:
            answer["fields"] = values
        return answer, sorted(set(reasons))

    def process(self, document):
        values, evidence = self.candidates(document)
        answer, reasons = self.decide(document, values, evidence)
        return answer, {**evidence, "reasons": reasons}


def load_settings():
    policy_path = ROOT / "policy.json"
    model_path = ROOT / "model.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    model = json.loads(model_path.read_text(encoding="utf-8"))
    return policy, model


def main():
    parser = argparse.ArgumentParser(description="Postprocess Russian passport OCR")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--audit", type=Path, help="Optional per-document reasons; contains no field values")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files")
    if args.audit and args.audit.resolve() in (args.input.resolve(), args.output.resolve()):
        parser.error("Audit, input and output must be different files")
    policy, model = load_settings()
    processor = Processor(policy, model)
    answers, audit, seen = [], [], set()
    with args.input.open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            document = json.loads(line)
            if document["doc_id"] in seen:
                raise ValueError(f"Duplicate doc_id on line {line_no}")
            seen.add(document["doc_id"])
            answer, evidence = processor.process(document)
            answers.append(answer)
            audit.append({"doc_id": document["doc_id"], "decision": answer["decision"],
                          "reasons": evidence["reasons"], "corrections": evidence["corrections"]})
    args.output.write_text("".join(json.dumps(a, ensure_ascii=False) + "\n" for a in answers), encoding="utf-8", newline="\n")
    if args.audit:
        args.audit.write_text("".join(json.dumps(a, ensure_ascii=False) + "\n" for a in audit), encoding="utf-8", newline="\n")
    print(json.dumps({"documents": len(answers), "accepted": sum(a["decision"] == "accept" for a in answers),
                      "as_of": AS_OF.isoformat()}, ensure_ascii=False), file=sys.stderr)


if __name__ == "__main__":
    main()
