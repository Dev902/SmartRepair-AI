"""Small local smoke test for SmartRepair's bundled data and prediction model."""
from pathlib import Path
import sys
import joblib
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "cleaned_repairs_en.csv"
CATEGORIES = ROOT / "cleaned_categories (1).csv"
MODEL = ROOT / "best_repair_model.joblib"
CLUSTERS = ROOT / "member3_clustered_repairs.csv"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)
    print("PASS:", message)


def main():
    require(DATA.exists(), "cleaned_repairs_en.csv exists")
    df = pd.read_csv(DATA, low_memory=False)
    required = {"product_category_id", "brand", "repair_status", "event_date", "problem", "problem_en"}
    require(required.issubset(df.columns), "cleaned dataset has the required app columns")
    require(CATEGORIES.exists(), "cleaned_categories (1).csv exists")
    names = pd.read_csv(CATEGORIES).set_index("product_category_id")["product_category"]
    require(not df.empty and df["product_category_id"].map(names).notna().all(), "dataset is non-empty and every category id has a name")

    require(MODEL.exists(), "best_repair_model.joblib exists")
    model = joblib.load(MODEL)
    require(sklearn.__version__ == "1.8.0", f"scikit-learn version is 1.8.0 (found {sklearn.__version__})")
    dates = pd.to_datetime(df["event_date"], errors="coerce")
    first = df.iloc[0]
    values = {
        "product_age": pd.to_numeric(pd.Series([first.get("product_age")]), errors="coerce").iloc[0],
        "year_of_manufacture": pd.to_numeric(pd.Series([first.get("year_of_manufacture")]), errors="coerce").iloc[0],
        "event_year": dates.dt.year.iloc[0],
        "event_month": dates.dt.month.iloc[0],
        "product_category_id": str(first.get("product_category_id", "Unknown")),
        "brand": str(first.get("brand", "Unknown")) if pd.notna(first.get("brand")) else "Unknown",
        "country": str(first.get("country", "Unknown")) if pd.notna(first.get("country")) else "Unknown",
        "data_provider": str(first.get("data_provider", "Unknown")) if pd.notna(first.get("data_provider")) else "Unknown",
    }
    input_df = pd.DataFrame([{c: values.get(c) for c in model.feature_names_in_}])
    for col in ["product_age", "year_of_manufacture", "event_year", "event_month"]:
        if col in input_df:
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce")
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    require(prediction in set(model.classes_), "model returns a known repair-status class")
    require(abs(float(probabilities.sum()) - 1.0) < 1e-6, "predicted class probabilities sum to 1")

    require((ROOT / "member2_model_comparison.csv").exists(), "three-model comparison file exists")
    require(pd.read_csv(ROOT / "member2_model_comparison.csv").shape[0] == 3, "three classifier results are present")
    require((ROOT / "model_confusion_matrix.csv").exists(), "bundled model confusion matrix exists")
    require(CLUSTERS.exists(), "Member 3 cluster file exists")
    cluster_columns = set(pd.read_csv(CLUSTERS, nrows=2, low_memory=False).columns)
    require({"fault_cluster", "repair_status", "problem_en"}.issubset(cluster_columns), "cluster output has cluster, outcome and description fields")
    print(f"\nSmoke test complete. Example predicted status: {prediction}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("FAIL:", exc, file=sys.stderr)
        sys.exit(1)
