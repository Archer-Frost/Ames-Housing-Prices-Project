from pathlib import Path
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LassoCV,Ridge
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor

from preprocessing import preprocessor
from feature_engineering import AmesFeatureEngineer

import random

random.seed(42)
np.random.seed(42)


def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "train_cleaned.csv"
    df = pd.read_csv(DATA_PATH)

    X_train = df.drop(columns=["SalePrice"])
    Y_train = np.log1p(df["SalePrice"])

    lasso_model = LassoCV(cv=5, max_iter=20000, random_state=42,n_jobs=-1)
    xgb_model = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        gamma=0.0,
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )

    lasso_pipe = Pipeline([
        ("fe", AmesFeatureEngineer()),
        ("pre", preprocessor(X_train)),
        ("lasso", lasso_model),
    ])

    xgb_pipe = Pipeline([
        ("fe", AmesFeatureEngineer()),
        ("pre", preprocessor(X_train)),
        ("xgb", xgb_model),
    ])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    stack = StackingRegressor(
        estimators=[
            ("xgb", xgb_pipe),
            ("lasso", lasso_pipe)
        ],
        final_estimator=Ridge(alpha=1.0),
        cv=kf,
        n_jobs=-1,
    )

    scores = -cross_val_score(
        stack,
        X_train,
        Y_train,
        cv=kf,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    print("RMSE mean:", scores.mean())
    print("RMSE std :", scores.std())

    stack.fit(X_train,Y_train)
    os.makedirs("models",exist_ok = True)
    joblib.dump(stack,"models/stack_model.joblib")
    print("Saved: models/stack_model.joblib")

if __name__ == "__main__":
    main()
