
"""
LoanLines: Predicting Loan Repayment Ability Using Machine Learning Algorithms

Dataset:
    Kaggle Home Credit Default Risk

Expected files inside ./data:
    application_train.csv
    bureau.csv
    bureau_balance.csv
    previous_application.csv
    POS_CASH_balance.csv
    installments_payments.csv
    credit_card_balance.csv

Target:
    TARGET = 0 -> Repay / non-default
    TARGET = 1 -> Default

Models taken from the attached research paper:
    Logistic Regression
    Random Forest
    Gaussian Naive Bayes
    LightGBM
    MLP Neural Network

The paper also uses class-imbalance handling, feature importance and
K-Means subgrouping. This implementation includes those parts.

Important engineering choice:
The Home Credit secondary tables contain multiple rows per borrower.
They are aggregated to SK_ID_CURR before merging so that each borrower
has exactly one training row. This avoids accidental row multiplication.
"""

from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
except ImportError:
    raise ImportError(
        "LightGBM is required. Run: pip install lightgbm"
    )

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("loanlines_output")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

# MLP is the most expensive model here. A limit makes the project
# practical on a normal student laptop. Set to None for full data.
MLP_MAX_TRAIN_ROWS = 60_000
RF_MAX_TRAIN_ROWS = 80_000

# The paper reports downsampling as its best imbalance strategy.
USE_UNDERSAMPLING = True

# The paper also experiments with K-Means borrower subgroups.
RUN_KMEANS_EXTENSION = True
KMEANS_K = 4


# ============================================================
# LOAD DATA
# ============================================================

def load_csv(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing file: {path}\n"
            f"Download the Home Credit Default Risk dataset and put "
            f"the CSV files inside the 'data' folder."
        )

    print(f"Loading {filename} ...")
    return pd.read_csv(path)


def load_home_credit_data():
    return {
        "application": load_csv("application_train.csv"),
        "bureau": load_csv("bureau.csv"),
        "bureau_balance": load_csv("bureau_balance.csv"),
        "previous": load_csv("previous_application.csv"),
        "pos_cash": load_csv("POS_CASH_balance.csv"),
        "installments": load_csv("installments_payments.csv"),
        "credit_card": load_csv("credit_card_balance.csv"),
    }


# ============================================================
# TABLE AGGREGATION
# ============================================================

def fast_aggregate(df, id_col="SK_ID_CURR", prefix=""):
    """Fast borrower-level aggregation designed for a normal laptop.

    Numeric columns: mean, min, max, count.
    Categorical columns: nunique, count.
    """
    if id_col not in df.columns:
        raise ValueError(f"{id_col} not found in table")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != id_col]

    categorical_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c != id_col]

    pieces = []

    if numeric_cols:
        numeric = df.groupby(id_col, sort=False)[numeric_cols].agg(
            ["mean", "min", "max", "count"]
        )
        numeric.columns = [
            f"{prefix}{col}_{stat}"
            for col, stat in numeric.columns
        ]
        pieces.append(numeric.reset_index())

    if categorical_cols:
        categorical = df.groupby(id_col, sort=False)[categorical_cols].agg(
            ["nunique", "count"]
        )
        categorical.columns = [
            f"{prefix}{col}_{stat}"
            for col, stat in categorical.columns
        ]
        pieces.append(categorical.reset_index())

    if not pieces:
        return df[[id_col]].drop_duplicates()

    result = pieces[0]
    for piece in pieces[1:]:
        result = result.merge(piece, on=id_col, how="outer")
    return result


def aggregate_table(df, prefix):
    return fast_aggregate(df, id_col="SK_ID_CURR", prefix=prefix)


def aggregate_bureau_balance(data):
    """Aggregate bureau_balance to bureau level, then to borrower level."""
    bureau = data["bureau"].copy()
    bb = data["bureau_balance"].copy()

    print("    Aggregating bureau_balance by SK_ID_BUREAU ...")
    bb_agg = fast_aggregate(
        bb,
        id_col="SK_ID_BUREAU",
        prefix="BB_"
    )
    del bb

    print("    Merging bureau_balance into bureau ...")
    bureau = bureau.merge(
        bb_agg,
        on="SK_ID_BUREAU",
        how="left"
    )
    del bb_agg

    print("    Aggregating bureau by SK_ID_CURR ...")
    result = fast_aggregate(
        bureau,
        id_col="SK_ID_CURR",
        prefix="BUREAU_"
    )
    del bureau
    return result


def prepare_dataset(data):
    application = data["application"].copy()

    if "TARGET" not in application.columns:
        raise ValueError(
            "TARGET column was not found in application_train.csv"
        )

    y = application["TARGET"].astype(int)

    X = application.drop(
        columns=["TARGET"]
    ).drop_duplicates(
        subset=["SK_ID_CURR"]
    )

    print("Aggregating bureau + bureau_balance ...")
    bureau = aggregate_bureau_balance(data)

    print("Aggregating previous applications ...")
    previous = aggregate_table(
        data["previous"],
        prefix="PREV_"
    )

    print("Aggregating POS/CASH ...")
    pos_cash = aggregate_table(
        data["pos_cash"],
        prefix="POS_"
    )

    print("Aggregating installments ...")
    installments = aggregate_table(
        data["installments"],
        prefix="INST_"
    )

    print("Aggregating credit-card history ...")
    credit_card = aggregate_table(
        data["credit_card"],
        prefix="CC_"
    )

    for table in [
        bureau,
        previous,
        pos_cash,
        installments,
        credit_card,
    ]:
        X = X.merge(
            table,
            on="SK_ID_CURR",
            how="left"
        )

    # Applicant ID is an identifier, not a useful ML feature.
    if "SK_ID_CURR" in X.columns:
        X = X.drop(columns=["SK_ID_CURR"])

    print(f"Rows: {len(X):,}")
    print(f"Raw feature count: {X.shape[1]:,}")

    return X, y


# ============================================================
# PREPROCESSING
# ============================================================

def clean_and_encode(X):
    X = X.copy()

    # Invalid infinite values become missing.
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if categorical_cols:
        print(
            f"One-hot encoding "
            f"{len(categorical_cols)} categorical columns ..."
        )

        X = pd.get_dummies(
            X,
            columns=categorical_cols,
            dummy_na=True
        )

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Reduce RAM usage.
    for col in X.columns:
        if pd.api.types.is_float_dtype(X[col]):
            X[col] = X[col].astype(np.float32)
        elif pd.api.types.is_integer_dtype(X[col]):
            X[col] = X[col].astype(np.int32)

    return X


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )


# ============================================================
# CLASS IMBALANCE
# ============================================================

def undersample_training_data(X_train, y_train):
    """
    Randomly undersample the majority class to create a 1:1
    training set. The test set is NOT undersampled.
    """
    temp = X_train.copy()
    temp["_TARGET_"] = y_train.to_numpy()

    class_0 = temp[temp["_TARGET_"] == 0]
    class_1 = temp[temp["_TARGET_"] == 1]

    minority_count = min(
        len(class_0),
        len(class_1)
    )

    class_0 = class_0.sample(
        n=minority_count,
        random_state=RANDOM_STATE
    )

    class_1 = class_1.sample(
        n=minority_count,
        random_state=RANDOM_STATE
    )

    balanced = pd.concat(
        [class_0, class_1]
    ).sample(
        frac=1,
        random_state=RANDOM_STATE
    )

    y_balanced = balanced["_TARGET_"].astype(int)
    X_balanced = balanced.drop(
        columns=["_TARGET_"]
    )

    return X_balanced, y_balanced


# ============================================================
# MODELS
# ============================================================

def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE
                )
            ),
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),

        "Gaussian Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianNB()),
        ]),

        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1
        ),

        "MLP Neural Network": Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    batch_size=256,
                    learning_rate_init=0.001,
                    max_iter=25,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=RANDOM_STATE
                )
            ),
        ]),
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    name,
    model,
    X_train,
    y_train,
    X_test,
    y_test
):
    print("\n" + "=" * 70)
    print(f"TRAINING: {name}")
    print("=" * 70)

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(
            y_test,
            y_pred
        ),
        "Precision_Default": precision_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0
        ),
        "Recall_Default": recall_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0
        ),
        "F1_Default": f1_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0
        ),
        "ROC_AUC": (
            roc_auc_score(y_test, y_prob)
            if y_prob is not None
            else np.nan
        ),
    }

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Repay / Non-default",
                "Default"
            ],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    print("\nMetrics:")
    for key, value in metrics.items():
        if key != "Model":
            print(f"{key}: {value:.4f}")

    return model, metrics


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def save_feature_importance(
    model_name,
    model,
    feature_names
):
    estimator = model

    if hasattr(model, "named_steps"):
        estimator = model.named_steps["model"]

    if not hasattr(
        estimator,
        "feature_importances_"
    ):
        return

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": estimator.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    )

    filename = (
        model_name.lower()
        .replace(" ", "_")
        + "_feature_importance.csv"
    )

    importance.to_csv(
        OUTPUT_DIR / filename,
        index=False
    )

    print(f"\nTop 20 features - {model_name}:")
    print(
        importance.head(20).to_string(
            index=False
        )
    )


# ============================================================
# K-MEANS EXTENSION
# ============================================================

def run_kmeans_extension(
    X_train,
    y_train,
    X_test,
    y_test,
    best_model,
    k=4
):
    """
    The paper reports K-Means borrower subgrouping.
    Here K-Means creates borrower clusters and the cluster ID
    is added as an additional feature.
    """
    print("\n" + "=" * 70)
    print("K-MEANS EXTENSION")
    print("=" * 70)

    max_kmeans_rows = min(
        100_000,
        len(X_train)
    )

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    if len(X_train) > max_kmeans_rows:
        indices = rng.choice(
            len(X_train),
            size=max_kmeans_rows,
            replace=False
        )
        X_fit = X_train[indices]
    else:
        X_fit = X_train

    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(
        X_fit
    )

    kmeans = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10
    )

    kmeans.fit(X_fit_scaled)

    train_clusters = kmeans.predict(
        scaler.transform(X_train)
    )
    test_clusters = kmeans.predict(
        scaler.transform(X_test)
    )

    train_cluster_features = np.eye(k)[
        train_clusters
    ]
    test_cluster_features = np.eye(k)[
        test_clusters
    ]

    X_train_ext = np.hstack([
        X_train,
        train_cluster_features
    ])

    X_test_ext = np.hstack([
        X_test,
        test_cluster_features
    ])

    extended_model = clone(best_model)

    extended_model.fit(
        X_train_ext,
        y_train
    )

    prediction = extended_model.predict(
        X_test_ext
    )

    probability = (
        extended_model.predict_proba(
            X_test_ext
        )[:, 1]
        if hasattr(
            extended_model,
            "predict_proba"
        )
        else None
    )

    print(f"K-Means k = {k}")
    print(
        f"Accuracy : "
        f"{accuracy_score(y_test, prediction):.4f}"
    )
    print(
        f"Precision: "
        f"{precision_score(y_test, prediction, zero_division=0):.4f}"
    )
    print(
        f"Recall   : "
        f"{recall_score(y_test, prediction, zero_division=0):.4f}"
    )
    print(
        f"F1       : "
        f"{f1_score(y_test, prediction, zero_division=0):.4f}"
    )

    if probability is not None:
        print(
            f"ROC-AUC  : "
            f"{roc_auc_score(y_test, probability):.4f}"
        )

    joblib.dump(
        {
            "scaler": scaler,
            "kmeans": kmeans,
            "model": extended_model,
        },
        OUTPUT_DIR / "kmeans_extension.joblib"
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("LOANLINES")
    print("Predicting Loan Repayment Ability Using ML Algorithms")
    print("=" * 70)

    # 1. Load Home Credit data.
    data = load_home_credit_data()

    # 2. Merge/aggregate borrower history.
    X_raw, y = prepare_dataset(data)

    print("\nTarget distribution:")
    print(y.value_counts())

    print("\nTarget percentage:")
    print(
        (y.value_counts(normalize=True) * 100)
        .round(2)
    )

    # 3. Encode categorical features.
    X = clean_and_encode(
        X_raw
    )

    feature_names = X.columns.tolist()

    print(
        f"\nEncoded feature count: "
        f"{len(feature_names):,}"
    )

    # 4. Train/test split.
    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    print(
        f"\nTraining rows: {len(X_train):,}"
    )
    print(
        f"Testing rows : {len(X_test):,}"
    )

    # 5. Missing-value handling.
    imputer = SimpleImputer(
        strategy="median"
    )

    X_train_imputed = imputer.fit_transform(
        X_train
    )

    X_test_imputed = imputer.transform(
        X_test
    )

    # 6. Balance ONLY the training data.
    if USE_UNDERSAMPLING:
        X_train_df = pd.DataFrame(
            X_train_imputed,
            columns=feature_names
        )

        X_train_balanced, y_train_balanced = (
            undersample_training_data(
                X_train_df,
                y_train.reset_index(drop=True)
            )
        )

        X_train_final = (
            X_train_balanced.to_numpy(
                dtype=np.float32
            )
        )

        y_train_final = (
            y_train_balanced.to_numpy()
        )

        print(
            f"\nBalanced training rows: "
            f"{len(y_train_final):,}"
        )
        print(
            "Balanced target distribution:"
        )
        print(
            pd.Series(
                y_train_final
            ).value_counts()
        )
    else:
        X_train_final = (
            X_train_imputed.astype(
                np.float32
            )
        )
        y_train_final = (
            y_train.to_numpy()
        )

    X_test_final = (
        np.asarray(
            X_test_imputed,
            dtype=np.float32
        )
    )
    y_test_final = (
        y_test.to_numpy()
    )

    # 7. Train the models from the paper.
    models = build_models()

    results = []
    trained_models = {}

    for name, model in models.items():

        current_X_train = X_train_final
        current_y_train = y_train_final

        # Keep Random Forest practical on a student laptop.
        if (
            name == "Random Forest"
            and len(current_X_train) > RF_MAX_TRAIN_ROWS
        ):
            rng = np.random.RandomState(RANDOM_STATE)
            indices = rng.choice(
                len(current_X_train),
                size=RF_MAX_TRAIN_ROWS,
                replace=False
            )
            current_X_train = current_X_train[indices]
            current_y_train = current_y_train[indices]
            print(f"\nRandom Forest training sample: {RF_MAX_TRAIN_ROWS:,} rows")

        # Keep MLP practical on a student laptop.
        if (
            name == "MLP Neural Network"
            and MLP_MAX_TRAIN_ROWS is not None
            and len(current_X_train)
            > MLP_MAX_TRAIN_ROWS
        ):
            rng = np.random.RandomState(
                RANDOM_STATE
            )

            indices = rng.choice(
                len(current_X_train),
                size=MLP_MAX_TRAIN_ROWS,
                replace=False
            )

            current_X_train = (
                current_X_train[indices]
            )
            current_y_train = (
                current_y_train[indices]
            )

            print(
                f"\nMLP training sample: "
                f"{MLP_MAX_TRAIN_ROWS:,} rows"
            )

        trained_model, metrics = evaluate_model(
            name,
            model,
            current_X_train,
            current_y_train,
            X_test_final,
            y_test_final
        )

        trained_models[name] = trained_model
        results.append(metrics)

        save_feature_importance(
            name,
            trained_model,
            feature_names
        )

        joblib.dump(
            {
                "model": trained_model,
                "imputer": imputer,
                "feature_names": feature_names,
            },
            OUTPUT_DIR
            / (
                name.lower()
                .replace(" ", "_")
                + ".joblib"
            )
        )

    # 8. Compare all models.
    results_df = pd.DataFrame(
        results
    ).sort_values(
        by="F1_Default",
        ascending=False
    )

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    results_df.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False
    )

    # 9. Select the model with the best default-class F1.
    best_model_name = (
        results_df.iloc[0]["Model"]
    )
    best_model = trained_models[
        best_model_name
    ]

    print(
        f"\nBest model by Default F1: "
        f"{best_model_name}"
    )

    # 10. K-Means extension from the paper.
    if RUN_KMEANS_EXTENSION:
        try:
            run_kmeans_extension(
                X_train_final,
                y_train_final,
                X_test_final,
                y_test_final,
                best_model,
                k=KMEANS_K
            )
        except Exception as exc:
            print(
                "\nK-Means extension could not be completed:"
                f" {exc}"
            )

    # 11. Save project metadata.
    metadata = {
        "project": "LoanLines",
        "target": "TARGET",
        "class_0": "Repay / Non-default",
        "class_1": "Default",
        "models": list(models.keys()),
        "undersampling": USE_UNDERSAMPLING,
        "kmeans_k": KMEANS_K,
        "encoded_feature_count": len(feature_names),
        "best_model_by_default_f1": best_model_name,
    }

    with open(
        OUTPUT_DIR / "metadata.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4
        )

    print("\n" + "=" * 70)
    print("LOANLINES TRAINING COMPLETE")
    print("=" * 70)
    print(
        f"Output folder: "
        f"{OUTPUT_DIR.resolve()}"
    )
    print("\nGenerated files include:")
    print("  model_comparison.csv")
    print("  *_feature_importance.csv")
    print("  *_regression.joblib / *_forest.joblib / etc.")
    print("  kmeans_extension.joblib")
    print("  metadata.json")


if __name__ == "__main__":
    main()
