# MLLoopOptSelector/src/workflow.py
"""
End-to-end: collect -> migrate_features -> train (try_models + regret weights + calibration + rare-label drop)
-> LOKO -> CV -> visuals (3D, SHAP)
-> GNN path (graphify + train_gnn, optional/auto-skip)
-> export (C header, md report).
"""
import subprocess, sys, os, argparse

ART = "artifacts"

def run(cmd):
    print("+", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)

def try_run(cmd, why=""):
    """Best-effort step that won't kill the workflow on failure."""
    print("+ (best-effort)", " ".join(cmd), ("-- " + why if why else ""))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"[warn] Skipped ({why or 'best-effort'}) with returncode={r.returncode}")

def have_torch():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser()
    # GNN options
    ap.add_argument("--skip-gnn", action="store_true",
                    help="Skip the GNN branch (graphify + train_gnn).")
    ap.add_argument("--gnn-epochs", type=int, default=100,
                    help="Epochs for GNN training.")
    ap.add_argument("--gnn-hidden", type=int, default=96,
                    help="Hidden size for GNN.")

    # LOKO options (forwarded to eval_loko.py)
    ap.add_argument("--loko-topk", type=int, default=5,
                help="Hybrid verify top-K candidates for LOKO (1 = off).")
    ap.add_argument("--loko-min-conf", type=float, default=0.80,
                    help="Confidence threshold to trust top-1 in LOKO; else verify.")
    ap.add_argument("--loko-always-include", type=str, default="baseline,omp,tile,tile_unroll",
                    help="Comma-separated labels always considered during LOKO verify.")
    ap.add_argument("--loko-force-family", type=str, default="RandomForest",
                    help="Force model family in LOKO (e.g. RandomForest, ExtraTrees, GBDT, LogReg, DecisionTree).")

    args = ap.parse_args()

    os.makedirs(ART, exist_ok=True)

    # 1) data collection
    run([sys.executable, "src/collect_data.py",
         "--sizes","128","192","256","320","384","448","512",
         "--repeats","4","--parallel","--jobs","4","--trim_outliers"])

    # 1.1) migrate/ensure engineered features exist (idempotent)
    run([sys.executable, "src/migrate_dataset_features.py",
         "--csv_in",  f"{ART}/dataset.csv",
         "--csv_out", f"{ART}/dataset.csv"])

    # 2) training (regret-based selection, calibrated, drop ultra-rare labels)
    run([sys.executable, "src/train_model.py",
         "--data_csv",  f"{ART}/dataset.csv",
         "--model_out", f"{ART}/model.joblib",
         "--report_out",f"{ART}/report.json",
         "--cost_weighting","--alpha","1.0","--cap","5.0",
         "--try_models",
         "--select_by","f1",
         "--calibrate",
         "--min_class_count","8",
         "--kfold","5"])

    # 3) LOKO (per-kernel regret/coverage/speedup) with hybrid verify + optional family override
    loko_cmd = [sys.executable, "src/eval_loko.py",
                "--data_csv", f"{ART}/dataset.csv",
                "--out_json", f"{ART}/loko.json",
                "--topk", str(args.loko_topk),
                "--min_conf", str(args.loko_min_conf),
                "--always_include", args.loko_always_include]
    if args.loko_force_family:
        loko_cmd += ["--force_family", args.loko_force_family]
    run(loko_cmd)

    # 4) Cross-validation (consistent label threshold)
    run([sys.executable, "src/cv_eval.py",
         "--data_csv", f"{ART}/dataset.csv",
         "--k","5",
         "--min_class_count","8",
         "--out_json", f"{ART}/cv.json"])

    # 5) Visuals & explainability
    run([sys.executable, "src/visualize_3d.py",
         "--data_csv", f"{ART}/dataset.csv",
         "--out_png",  f"{ART}/tile_runtime_3d.png"])

    run([sys.executable, "src/explain_choice.py",
         "--data_csv", f"{ART}/dataset.csv",
         "--model_in", f"{ART}/model.joblib",
         "--out_png",  f"{ART}/explain.png"])

    # 5.5) GNN path (optional): graphify + train_gnn
    # Auto-skip if --skip-gnn or torch is missing.
    if not args.skip_gnn and have_torch():
        # Build graphs from the dataset for each group_id
        run([sys.executable, "src/graphify.py",
             "--csv",     f"{ART}/dataset.csv",
             "--out_pt",  f"{ART}/gnn_graphs.pt",
             "--out_meta",f"{ART}/gnn_meta.json"])

        # Train the GNN
        run([sys.executable, "src/train_gnn.py",
             "--graphs_pt", f"{ART}/gnn_graphs.pt",
             "--epochs",    str(args.gnn_epochs),
             "--hidden",    str(args.gnn_hidden),
             "--out_model", f"{ART}/gnn_model.pt",
             "--out_report",f"{ART}/gnn_report.json"])

        # After training the GNN, predict all + regret
        run([sys.executable, "src/gnn_predict_all.py",
             "--graphs_pt", f"{ART}/gnn_graphs.pt",
             "--model_pt",  f"{ART}/gnn_model.pt",
             "--out_csv",   f"{ART}/gnn_pred.csv"])

        run([sys.executable, "src/evaluate_regret_gnn.py",
             "--data_csv",   f"{ART}/dataset.csv",
             "--pred_csv",   f"{ART}/gnn_pred.csv",
             "--summary_out",f"{ART}/gnn_regret_summary.json",
             "--details_out",f"{ART}/gnn_regret_details.csv"])
    else:
        reason = "--skip-gnn flag" if args.skip_gnn else "torch not installed"
        print(f"[info] Skipping GNN branch ({reason}).")

    # 6) Export C header + Markdown report
    run([sys.executable, "src/export_header.py",
         "--data_csv", f"{ART}/dataset.csv",
         "--out_h",    f"{ART}/generated_schedule.h"])

    run([sys.executable, "src/report.py",
         "--out_md",   f"{ART}/report.md"])

    print("\nWorkflow complete. See artifacts/ for outputs:")
    print(" - dataset.csv, report.json/txt, cv.json, loko.json")
    print(" - model.joblib, feature_importance.csv, explain.png, tile_runtime_3d.png")
    if not args.skip_gnn:
        print(" - gnn_graphs.pt, gnn_model.pt, gnn_report.json (if torch available)")
    print(" - generated_schedule.h, report.md")

if __name__ == "__main__":
    main()
