# MLLoopOptSelector/src/plot_confusion.py
import argparse, json, pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report_json", type=str, default="artifacts/report.json")
    ap.add_argument("--out_png", type=str, default="artifacts/confusion_matrix.png")
    args = ap.parse_args()

    with open(args.report_json) as f:
        rep = json.load(f)
    labels = rep["classes"]
    cm = rep["confusion_matrix"]

    fig = plt.figure()
    ax = fig.gca()
    im = ax.imshow(cm)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i][j], ha="center", va="center")
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=160)
    print(f"Saved {args.out_png}")

if __name__ == "__main__":
    main()
