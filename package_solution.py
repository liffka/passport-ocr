import ast
import hashlib
import io
import json
from pathlib import Path
import tokenize
import zipfile

ROOT = Path(__file__).resolve().parent
FILES = (
    ".gitattributes", ".gitignore", "README.md", "REPORT.md", "THIRD_PARTY_NOTICES.md",
    "postprocess.py", "fit_model.py", "run_cv.py", "cv_tools.py", "validate.py",
    "evaluate.py", "forecast.py", "check_delivery.py", "test_postprocess.py", "test_cv.py",
    "build_report.py", "package_solution.py", "policy.json", "model.json", "answers.jsonl",
    "reference/subdivisions.csv", "reference/SOURCE.txt", "output/pdf/report.pdf",
    "validation/cv_final/protocol.json", "validation/cv_final/summary.json",
    "validation/cv_final/RESULTS.md", "validation/train_metrics.json",
    "validation/forecast.json", "validation/delivery_check.json",
)


def check_python(path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    if any(isinstance(node, (ast.Try, ast.TryStar)) for node in ast.walk(tree)):
        raise ValueError(f"Unexpected try/except in {path.name}")
    if any(token.type == tokenize.COMMENT
           for token in tokenize.generate_tokens(io.StringIO(source).readline)):
        raise ValueError(f"Unexpected comment in {path.name}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if ast.get_docstring(node):
                raise ValueError(f"Unexpected docstring in {path.name}")


def main():
    summary = json.loads((ROOT / "validation/cv_final/summary.json").read_text(encoding="utf-8"))
    forecast = json.loads((ROOT / "validation/forecast.json").read_text(encoding="utf-8"))
    files = list(FILES)
    files.extend(f"validation/cv_final/grouped_cv/repeat-{repeat}/metrics.json"
                 for repeat in range(summary["protocol"]["repeats"]))
    for name in files:
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(name)
        if path.suffix == ".py":
            check_python(path)

    manifest = {
        "version": "submission",
        "as_of": "2026-10-01",
        "test_accepted": forecast["test_accepted"],
        "test_rejected": forecast["test_rejected"],
        "expected_errors": forecast["expected_errors"],
        "rounded_error_forecast": forecast["rounded_error_forecast"],
        "zero_error_claim": False,
        "sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                   for name in sorted(files)},
    }
    manifest_name = "validation/delivery_manifest.json"
    (ROOT / manifest_name).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8", newline="\n")
    destination = ROOT.parent / "passport-ocr-solution.zip"
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files + [manifest_name]):
            archive.write(ROOT / name, "passport-ocr-solution/" + name)
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        for name, expected in manifest["sha256"].items():
            content = archive.read("passport-ocr-solution/" + name)
            assert hashlib.sha256(content).hexdigest() == expected, name
    print(json.dumps({"archive": str(destination), "files": len(files) + 1,
                      "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
