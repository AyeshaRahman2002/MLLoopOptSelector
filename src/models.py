# MLLoopOptSelector/src/models.py
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def get_default_model(seed=42):
    return ExtraTreesClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1
        # class_weight handled via sample_weight upstream
    )

def get_model_zoo(seed=42):
    zoo = {
        "RandomForest": RandomForestClassifier(
            n_estimators=700,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=800,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
        "GBDT": GradientBoostingClassifier(random_state=seed),
        "LogReg": LogisticRegression(max_iter=10000, solver="saga", penalty="l2", n_jobs=-1),
        "DecisionTree": DecisionTreeClassifier(random_state=seed),
    }
    # Optional gradient-boosting libs
    try:
        from xgboost import XGBClassifier
        zoo["XGBoost"] = XGBClassifier(
            n_estimators=800, max_depth=6, subsample=0.8, colsample_bytree=0.8,
            learning_rate=0.05, reg_lambda=1.0, random_state=seed, n_jobs=-1,
            objective="multi:softprob"
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier
        zoo["LightGBM"] = LGBMClassifier(
            n_estimators=1000, num_leaves=96, learning_rate=0.05, random_state=seed
        )
    except Exception:
        pass
    # Optional neural option (requires torch)
    try:
        from nn_models import TorchMLPClassifier
        zoo["TorchMLP"] = TorchMLPClassifier(
            seed=seed, epochs=60, lr=1e-3, batch_size=256,
            hidden=(256, 128, 64), pdrop=0.10
        )
    except Exception:
        # torch not installed or file missing; ignore
        pass
    return zoo
