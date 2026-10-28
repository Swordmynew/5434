"""Export the saved text model as a reproducible single-model submission."""
import argparse
import csv
import hashlib
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default="runs/main/predictions/text/test.csv")
    parser.add_argument("--sample", default="data/raw/sampleSubmission.csv")
    parser.add_argument("--output", default="runs/main/submission_text_only.csv")
    parser.add_argument("--expected-rows", type=int, default=10000)
    args = parser.parse_args()
    with Path(args.sample).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["id", "rating"]:
            raise ValueError("Unexpected sample submission columns")
        ids = [r["id"] for r in reader]
    with Path(args.predictions).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["id", "prediction"]:
            raise ValueError("Unexpected component prediction columns")
        rows = list(reader)
    values = {r["id"]: float(r["prediction"]) for r in rows}
    if len(ids) != args.expected_rows or len(set(ids)) != len(ids):
        raise ValueError("Sample row count or unique IDs do not match")
    if len(values) != len(rows) or set(values) != set(ids):
        raise ValueError("Text prediction IDs do not match the sample")
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError("Non-finite predictions")
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError("Use a new output path to preserve an existing candidate")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "rating"])
        writer.writerows((ident, format(min(5., max(1., values[ident])), ".10f")) for ident in ids)
    temporary.replace(destination)
    print("Submission:", destination.resolve())
    print("Rows:", len(ids))
    print("SHA256:", hashlib.sha256(destination.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
