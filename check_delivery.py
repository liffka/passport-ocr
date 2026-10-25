from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from postprocess import FIELDS, ROOT, validity_reason


def main():
    with (ROOT / "data/test_predictions.jsonl").open(encoding="utf-8") as f:
        documents = list(map(json.loads, f))
    with (ROOT / "answers.jsonl").open(encoding="utf-8") as f:
        answers = list(map(json.loads, f))
    ids = [r["doc_id"] for r in answers]
    assert len(answers) == len(documents) == len(set(ids)) == 2000
    assert set(ids) == {r["doc_id"] for r in documents}
    for answer in answers:
        assert answer["decision"] in ("accept", "reject")
        if answer["decision"] == "accept":
            assert set(answer) == {"doc_id", "decision", "fields"}
            assert set(answer["fields"]) == set(FIELDS)
            assert all(isinstance(v, str) for v in answer["fields"].values())
            assert validity_reason(answer["fields"]) is None
        else:
            assert set(answer) == {"doc_id", "decision"}
    staging = ROOT / "tmp/offline-check"
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "reference").mkdir(exist_ok=True)
    for name in ("postprocess.py", "model.json", "policy.json", "reference/subdivisions.csv"):
        shutil.copyfile(ROOT / name, staging / name)
    assert not (staging / "data").exists()
    wrapper = """import runpy, socket, sys
def denied(*args, **kwargs):
    raise RuntimeError('Network access is disabled during the offline check')
socket.socket.connect = denied
socket.create_connection = denied
sys.argv = [sys.argv[1], sys.argv[2], sys.argv[3]]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    output = staging / "answers.jsonl"
    command = [sys.executable, "-I", "-S", "-c", wrapper,
               str(staging / "postprocess.py"), str(ROOT / "data/test_predictions.jsonl"), str(output)]
    subprocess.run(command, cwd=staging, check=True, capture_output=True)
    expected = hashlib.sha256((ROOT / "answers.jsonl").read_bytes()).hexdigest()
    actual = hashlib.sha256(output.read_bytes()).hexdigest()
    assert actual == expected, "Offline output differs from the delivered answers"
    result = {"documents": len(answers), "decisions": dict(Counter(r["decision"] for r in answers)),
              "answers_sha256": actual, "stdlib_only": True, "training_files_absent": True,
              "network_connect_blocked": True, "output_reproduced_exactly": True}
    (ROOT / "validation/delivery_check.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
