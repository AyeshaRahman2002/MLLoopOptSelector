# src/report.py
import argparse, json, pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_md", default="artifacts/report.md")
    ap.add_argument("--train_json", default="artifacts/report.json")
    ap.add_argument("--loko_json", default="artifacts/loko.json")
    ap.add_argument("--cv_json", default="artifacts/cv.json")
    args = ap.parse_args()

    Path("artifacts").mkdir(exist_ok=True)
    lines = ["# MLLoopOptSelector Report\n"]
    def add_json(title, path):
        if Path(path).exists():
            lines.append(f"## {title}\n")
            data = json.loads(Path(path).read_text())
            lines.append("```json\n" + json.dumps(data, indent=2) + "\n```\n")
    add_json("Training", args.train_json)
    add_json("LOKO", args.loko_json)
    add_json("Cross-Validation", args.cv_json)
    # --- Add GNN sections if present ---
    add_json("GNN (Training)", "artifacts/gnn_report.json")
    add_json("GNN (Regret Summary)", "artifacts/gnn_regret_summary.json")
    
    # Include any cross-arch evals if they exist (convention: artifacts/xarch*.json)
    from glob import glob
    for path in sorted(glob("artifacts/xarch*.json")):
        add_json(f"Cross-Architecture ({path.split('/')[-1]})", path)

    Path(args.out_md).write_text("\n".join(lines))
    print(f"Wrote {args.out_md}")

if __name__ == "__main__":
    main()
