PY=python3
ART=artifacts

.PHONY: all smoke data train predict clean deepclean distclean loko hybrid-demo tune export quick \
        cv models vis3d explain meta fewshot workflow report profile collect-fast \
        gnn gnn-data gnn-train gnn-predict

all: data train

smoke:
	$(PY) src/collect_data.py --sizes 128 256 --repeats 2

data:
	$(PY) src/collect_data.py --sizes 128 192 256 320 384 448 512 --repeats 3

train:
	$(PY) src/train_model.py --data_csv $(ART)/dataset.csv --model_out $(ART)/model.joblib --report_out $(ART)/report.json

predict:
	$(PY) src/predict.py --kernel matmul --N 384 --M 384 --K 384

quick:
	$(PY) src/collect_data.py --sizes 128 256 --repeats 2
	$(PY) src/train_model.py --data_csv $(ART)/dataset.csv --model_out $(ART)/model.joblib --report_out $(ART)/report.json

# Safe: remove generated files but keep the artifacts/ folder
clean:
	@echo "[clean] removing generated files in $(ART)"
	@rm -f $(ART)/*.csv \
	        $(ART)/*.csv.bak \
	        $(ART)/*.json \
	        $(ART)/*.txt \
	        $(ART)/*.png \
	        $(ART)/*.h \
	        $(ART)/*.md \
	        $(ART)/*.joblib \
	        $(ART)/*.pt \
	        $(ART)/*_pred.csv \
	        $(ART)/*_regret_*.json \
	        $(ART)/*_regret_*.csv
	@rm -f $(ART)/feature_importance.csv

# Stronger: also remove subdirs (models_meta), caches, stray pyc
deepclean: clean
	@echo "[deepclean] removing subdirectories & caches"
	@rm -rf $(ART)/models_meta
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	@find . -name "*.pyc" -delete

# Nuclear: remove the entire artifacts directory
distclean:
	@echo "[distclean] removing $(ART) directory"
	@rm -rf $(ART)

loko:
	$(PY) src/eval_loko.py --data_csv $(ART)/dataset.csv --out_json $(ART)/loko.json

hybrid-demo:
	$(PY) src/hybrid_select.py --kernel matmul --N 384 --M 384 --K 384 --topk 2 --min_conf 0.8 --hybrid auto

tune:
	$(PY) src/tune_tiles.py --kernel matmul --N 384 --M 384 --K 384 --random

export:
	$(PY) src/export_header.py --data_csv $(ART)/dataset.csv --out_h $(ART)/generated_schedule.h

# NEW/UPDATED:
cv:
	$(PY) src/cv_eval.py --data_csv $(ART)/dataset.csv --out_json $(ART)/cv.json

models:
	$(PY) src/train_model.py --data_csv $(ART)/dataset.csv --model_out $(ART)/model.joblib --report_out $(ART)/report.json --try_models

vis3d:
	$(PY) src/visualize_3d.py --data_csv $(ART)/dataset.csv --out_png $(ART)/tile_runtime_3d.png

explain:
	$(PY) src/explain_choice.py --data_csv $(ART)/dataset.csv --model_in $(ART)/model.joblib --out_png $(ART)/explain.png

meta:
	$(PY) src/train_meta.py --data_csv $(ART)/dataset.csv --out_dir $(ART)/models_meta

fewshot:
	$(PY) src/finetune_fewshot.py --base_model $(ART)/model.joblib --data_csv $(ART)/dataset.csv --hold_kernel conv1d --k 16 --out_model $(ART)/fewshot.joblib

workflow:
	$(PY) src/workflow.py

report:
	$(PY) src/report.py --out_md $(ART)/report.md

# fast collector with parallel build/run + noise controls
collect-fast:
	$(PY) src/collect_data.py --sizes 128 192 256 320 384 448 512 --repeats 5 --parallel --jobs 4 --trim_outliers

# Cross-architecture evaluation:
# usage:
#   make xarch MODEL=artifacts/models_meta/<archA>.joblib DATA=path/to/other_machine_dataset.csv OUT=artifacts/xarch_archA_on_archB.json
xarch:
	$(PY) src/eval_cross_arch.py --model_in $(MODEL) --data_csv $(DATA) --out_json $(OUT)

nn:
	$(PY) src/train_model.py --data_csv $(ART)/dataset.csv --model_out $(ART)/model.joblib --report_out $(ART)/report.json --try_models --select_by regret --cost_weighting --alpha 2 --cap 10 --calibrate --min_class_count 8


gnn-data:
	$(PY) src/graphify.py --csv $(ART)/dataset.csv --out_pt $(ART)/gnn_graphs.pt --out_meta $(ART)/gnn_meta.json

gnn-train:
	# remove --lr if your train_gnn.py doesn't accept it
	$(PY) src/train_gnn.py --graphs_pt $(ART)/gnn_graphs.pt --epochs 100 --hidden 96 --out_model $(ART)/gnn_model.pt --out_report $(ART)/gnn_report.json

gnn-predict:
	$(PY) src/gnn_predict_all.py --graphs_pt $(ART)/gnn_graphs.pt --model_pt $(ART)/gnn_model.pt --out_csv $(ART)/gnn_pred.csv

gnn:
	$(PY) src/graphify.py --csv $(ART)/dataset.csv --out_pt $(ART)/gnn_graphs.pt --out_meta $(ART)/gnn_meta.json
	$(PY) src/train_gnn.py --graphs_pt $(ART)/gnn_graphs.pt --epochs 100 --hidden 96 --out_model $(ART)/gnn_model.pt --out_report $(ART)/gnn_report.json
	$(PY) src/gnn_predict_all.py --graphs_pt $(ART)/gnn_graphs.pt --model_pt $(ART)/gnn_model.pt --out_csv $(ART)/gnn_pred.csv
	$(PY) src/evaluate_regret_gnn.py --data_csv $(ART)/dataset.csv --pred_csv $(ART)/gnn_pred.csv --summary_out $(ART)/gnn_regret_summary.json --details_out $(ART)/gnn_regret_details.csv