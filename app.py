"""SmartRepair AI: a simple Streamlit app for repair-record analysis."""
from datetime import datetime
from pathlib import Path
import json
import re
import uuid

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Bundled files are kept in the same folder as this script.
FOLDER = Path(__file__).resolve().parent
DATA_FILE = FOLDER / "cleaned_repairs_en.csv"
CATEGORY_FILE = FOLDER / "cleaned_categories (1).csv"
MEMBER2_FILE = FOLDER / "member2_model_comparison.csv"
FAULT_FILE = FOLDER / "member3_clustered_repairs.csv"

# Records and models created from the app are saved here. The bundled files are never modified.
USER_DIR = FOLDER / "user_data"
USER_RECORDS = USER_DIR / "user_records.csv"
CLUSTER_ASSIGNER = USER_DIR / "fault_cluster_assigner.joblib"

# Model artifacts: a retrained copy in user_data/ takes priority over the bundled file.
MODEL_NAME = "best_repair_model.joblib"
METRICS_NAME = "model_comparison.csv"
MATRIX_NAME = "model_confusion_matrix.csv"
REPORT_NAME = "model_classification_report.csv"
INFO_NAME = "model_training_info.json"
MODEL_ARTIFACTS = [MODEL_NAME, METRICS_NAME, MATRIX_NAME, REPORT_NAME, INFO_NAME]

RECORD_COLUMNS = ["id", "data_provider", "country", "product_category_id", "brand", "year_of_manufacture", "product_age",
                  "repair_status", "repair_barrier", "event_date", "problem", "problem_en", "product_category"]
ORIGINAL_SOURCE = "Original dataset"
USER_SOURCE = "User added"
NUMERIC_FEATURES = ["product_age", "year_of_manufacture", "event_year", "event_month"]
CATEGORICAL_FEATURES = ["product_category_id", "brand", "country", "data_provider"]
TRAINING_SAMPLE = 60000

st.set_page_config(page_title="SmartRepair AI", layout="wide")
st.title("SmartRepair AI")
st.write("Explore repair records, repair outcomes, common faults and demand trends.")


def version(path):
    """Changes whenever the file changes, so cached results are refreshed automatically."""
    return path.stat().st_mtime_ns if path.exists() else 0


def artifact(name):
    user_file = USER_DIR / name
    return user_file if user_file.exists() else FOLDER / name


def using_retrained_model():
    return (USER_DIR / MODEL_NAME).exists()


@st.cache_resource
def read_model(path, file_version):
    return joblib.load(path)


def prepare_data(data):
    """Use consistent column names and prepare dates/numbers for charts."""
    data = data.copy()
    if "repair_barrier" not in data and "repair_barrier_if_end_of_life" in data:
        data["repair_barrier"] = data["repair_barrier_if_end_of_life"]
    if "product_category" not in data and "partner_product_category" in data:
        data["product_category"] = data["partner_product_category"]
    if "product_category" not in data:
        ids = data.get("product_category_id", pd.Series("Unknown", index=data.index))
        data["product_category"] = "Category ID " + ids.fillna("Unknown").astype(str)
    if "product_category_id" in data:
        data["product_category_id"] = pd.to_numeric(data["product_category_id"], errors="coerce").astype("Int64")
    if "event_date" in data:
        data["event_date"] = pd.to_datetime(data["event_date"], errors="coerce")
        data["event_year"] = data["event_date"].dt.year
        data["event_month"] = data["event_date"].dt.month
    for col in ["product_age", "year_of_manufacture"]:
        if col in data:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


@st.cache_data(max_entries=2, show_spinner="Loading repair dataset...")
def load_dataset(base_version, category_version, user_version):
    """Bundled dataset plus any records added from the app."""
    data = pd.read_csv(DATA_FILE, low_memory=False)
    if CATEGORY_FILE.exists():
        names = pd.read_csv(CATEGORY_FILE).set_index("product_category_id")["product_category"]
        data["product_category"] = data["product_category_id"].map(names)
    data["source"] = ORIGINAL_SOURCE
    if USER_RECORDS.exists():
        user = pd.read_csv(USER_RECORDS, low_memory=False)
        user["source"] = USER_SOURCE
        data = pd.concat([data, user], ignore_index=True)
    return prepare_data(data)


def read_user_records():
    if USER_RECORDS.exists():
        return pd.read_csv(USER_RECORDS, low_memory=False)
    return pd.DataFrame(columns=RECORD_COLUMNS)


def write_user_records(records):
    USER_DIR.mkdir(exist_ok=True)
    if records.empty:
        USER_RECORDS.unlink(missing_ok=True)
    else:
        records[RECORD_COLUMNS].to_csv(USER_RECORDS, index=False)


def normalize_records(raw, existing):
    """Convert user-supplied rows to the dataset's columns. Returns (valid records, notes for the user)."""
    data = raw.copy()
    data.columns = [str(c).strip().lower().replace(" ", "_") for c in data.columns]
    data = data.rename(columns={"repair_barrier_if_end_of_life": "repair_barrier", "category": "product_category",
                                "status": "repair_status", "date": "event_date"})
    data = data.loc[:, ~data.columns.duplicated()]
    notes = []
    if "problem_en" not in data and "problem" in data:
        notes.append("No problem_en column: the problem text is used as the English description.")
    missing = [c for c in ["repair_status", "event_date"] if c not in data]
    if "product_category" not in data and "product_category_id" not in data:
        missing.append("product_category")
    if missing:
        return pd.DataFrame(columns=RECORD_COLUMNS), [f"Missing required column(s): {', '.join(missing)}."]
    for col in RECORD_COLUMNS:
        if col not in data:
            data[col] = np.nan
    data = data[RECORD_COLUMNS].copy()
    for col in data:
        if data[col].dtype == object or pd.api.types.is_string_dtype(data[col]):
            data[col] = data[col].astype("string").str.strip().replace("", pd.NA).astype(object)

    # Match category names/IDs to the existing ones so new rows join the same groups.
    known = existing.dropna(subset=["product_category_id"]).drop_duplicates("product_category")
    id_to_name = known.drop_duplicates("product_category_id").set_index("product_category_id")["product_category"]
    name_lookup = {str(n).lower(): n for n in known["product_category"]}
    name_to_id = dict(zip(known["product_category"], known["product_category_id"]))
    ids = pd.to_numeric(data["product_category_id"], errors="coerce").astype("Int64")
    names = data["product_category"].map(lambda n: name_lookup.get(str(n).lower(), n) if pd.notna(n) else n)
    names = names.fillna(ids.map(id_to_name))
    data["product_category"] = names
    data["product_category_id"] = ids.fillna(names.map(name_to_id).astype("Int64"))

    status_lookup = {str(s).lower(): s for s in existing["repair_status"].dropna().unique()}
    data["repair_status"] = data["repair_status"].map(lambda s: status_lookup.get(str(s).lower(), s) if pd.notna(s) else s)
    dates = pd.to_datetime(data["event_date"], errors="coerce")
    data["event_date"] = dates.dt.strftime("%Y-%m-%d")
    for col in ["year_of_manufacture", "product_age"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in ["brand", "country"]:
        data[col] = data[col].map(lambda v: str(v).upper() if pd.notna(v) else v)
    data["data_provider"] = data["data_provider"].fillna("SmartRepair user")
    data["problem_en"] = data["problem_en"].fillna(data["problem"])
    data["problem"] = data["problem"].fillna(data["problem_en"])

    invalid = dates.isna() | data["repair_status"].isna() | data["product_category"].isna()
    if invalid.any():
        notes.append(f"{invalid.sum():,} row(s) skipped because event_date, repair_status or product_category is missing or invalid.")
    data = data[~invalid]
    duplicate = data["id"].notna() & (data["id"].astype(str).isin(set(existing["id"].astype(str))) | data["id"].duplicated())
    if duplicate.any():
        notes.append(f"{duplicate.sum():,} row(s) skipped because their id is already in the dataset.")
    data = data[~duplicate].copy()
    new_statuses = sorted(set(data["repair_status"].astype(str)) - set(map(str, status_lookup.values())))
    if new_statuses:
        notes.append(f"New repair status value(s) will be added: {', '.join(new_statuses)}.")
    new_categories = sorted(set(data["product_category"].astype(str)) - set(map(str, name_lookup.values())))
    if new_categories:
        notes.append(f"New product category value(s) will be added: {', '.join(new_categories)}.")
    data["id"] = data["id"].astype(object)
    no_id = data["id"].isna()
    data.loc[no_id, "id"] = [f"user_{uuid.uuid4().hex[:12]}" for _ in range(no_id.sum())]
    return data.reset_index(drop=True), notes


def clean_text(text):
    """Same text cleaning as the Member 3 fault-analysis notebook."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_resource(show_spinner="Preparing the fault-cluster model (first time only)...")
def cluster_assigner(file_version):
    """TF-IDF settings from the Member 3 notebook plus the centre of each exported cluster."""
    if CLUSTER_ASSIGNER.exists():
        saved = joblib.load(CLUSTER_ASSIGNER)
        if saved.get("file_version") == file_version:
            return saved
    clusters = pd.read_csv(FAULT_FILE, usecols=["clean_problem", "fault_cluster"]).dropna()
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.95, max_features=10000)
    matrix = tfidf.fit_transform(clusters["clean_problem"])
    labels = np.sort(clusters["fault_cluster"].unique())
    centres = np.vstack([np.asarray(matrix[(clusters["fault_cluster"] == k).to_numpy()].mean(axis=0)) for k in labels])
    saved = {"file_version": file_version, "tfidf": tfidf, "centres": centres, "labels": labels}
    USER_DIR.mkdir(exist_ok=True)
    joblib.dump(saved, CLUSTER_ASSIGNER)
    return saved


def assign_clusters(texts):
    """Put new descriptions in the nearest existing fault cluster (as K-Means would)."""
    cleaned = texts.fillna("").map(clean_text)
    result = pd.Series(np.nan, index=texts.index)
    usable = cleaned.str.len() > 0
    if usable.any():
        model = cluster_assigner(version(FAULT_FILE))
        distances = euclidean_distances(model["tfidf"].transform(cleaned[usable]), model["centres"])
        result[usable] = model["labels"][distances.argmin(axis=1)]
    return result


@st.cache_data(max_entries=2, show_spinner="Loading fault clusters...")
def load_fault_data(base_version, user_version, fault_version):
    data = load_dataset(base_version, version(CATEGORY_FILE), user_version)
    clusters = pd.read_csv(FAULT_FILE, usecols=["id", "fault_cluster"])
    data = data.merge(clusters, on="id", how="left")
    added = data["source"] == USER_SOURCE
    if added.any():
        data.loc[added, "fault_cluster"] = assign_clusters(data.loc[added, "problem_en"].fillna(data.loc[added, "problem"]))
    data = data.dropna(subset=["fault_cluster"])
    data["fault_cluster"] = data["fault_cluster"].astype(int)
    return data


def retrain_model(data):
    """Retrain the Logistic Regression pipeline from the Member 2 notebook on the current dataset."""
    data = data.dropna(subset=["repair_status"]).copy()
    data["repair_status"] = data["repair_status"].astype(str).str.strip()
    data = data[data["repair_status"].ne("") & data["repair_status"].str.lower().ne("nan")]
    data = data[data["repair_status"].map(data["repair_status"].value_counts()) >= 10]

    # Keep every user-added record and fill the rest of the sample from the original data.
    added = data[data["source"] == USER_SOURCE]
    original = data[data["source"] != USER_SOURCE]
    original_size = max(0, TRAINING_SAMPLE - len(added))
    if original_size < len(original):
        original, _ = train_test_split(original, train_size=original_size, random_state=42, stratify=original["repair_status"])
    sample = pd.concat([original, added])

    X = sample[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = sample["repair_status"]
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("string").fillna("Unknown").astype(str)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUMERIC_FEATURES),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                          ("encoder", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_FEATURES),
    ])
    model = Pipeline([("preprocessor", preprocessor),
                      ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    USER_DIR.mkdir(exist_ok=True)
    joblib.dump(model, USER_DIR / MODEL_NAME)
    pd.DataFrame([{
        "Model": f"Logistic Regression (retrained, {len(sample):,} records incl. {len(added):,} user-added)",
        "Training records": len(X_train), "Test records": len(X_test),
        "Accuracy": accuracy_score(y_test, predicted),
        "Precision": precision_score(y_test, predicted, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, predicted, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, predicted, average="weighted", zero_division=0),
    }]).to_csv(USER_DIR / METRICS_NAME, index=False)
    labels = model.classes_
    pd.DataFrame(confusion_matrix(y_test, predicted, labels=labels), index=labels, columns=labels) \
        .rename_axis("Actual / Predicted").to_csv(USER_DIR / MATRIX_NAME)
    pd.DataFrame(classification_report(y_test, predicted, output_dict=True, zero_division=0)).T.to_csv(USER_DIR / REPORT_NAME)
    (USER_DIR / INFO_NAME).write_text(json.dumps({
        "model": "Logistic Regression",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "sample_size": len(sample), "user_added_records": len(added),
        "train_size": len(X_train), "test_size": len(X_test), "random_state": 42,
        "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES, "target": "repair_status",
        "notes": "Retrained from the app: all user-added records plus a stratified sample of the original dataset.",
    }, indent=2), encoding="utf-8")


def choices(data, column):
    if column not in data:
        return ["Unknown"]
    values = data[column].fillna("Unknown").astype(str).str.strip().unique().tolist()
    return sorted(v for v in values if v and v.lower() != "nan") or ["Unknown"]


def filter_data(data, categories, brand_text, statuses, sources):
    result = data.copy()
    if "product_category" in result:
        result = result[result["product_category"].fillna("Unknown").astype(str).isin(categories)]
    if brand_text.strip() and "brand" in result:
        result = result[result["brand"].fillna("Unknown").astype(str).str.contains(brand_text.strip(), case=False, regex=False)]
    if "repair_status" in result:
        result = result[result["repair_status"].fillna("Unknown").astype(str).isin(statuses)]
    if "source" in result:
        result = result[result["source"].isin(sources)]
    return result


def recommendation(status):
    status = str(status).lower()
    if "fixed" in status:
        return "Verify the repair before returning the product to the customer."
    if "repairable" in status:
        return "Inspect the product and estimate repair time and parts."
    if "end of life" in status or "end-of-life" in status:
        return "Review safety, cost and parts availability with the customer."
    return "Collect more information and review the case manually."


def forecast(monthly):
    """Compare simple forecasts on recent months and estimate the next six months."""
    monthly = monthly.asfreq("MS", fill_value=0).astype(float)
    if len(monthly) < 8:
        return None, None
    test_count = min(6, max(2, len(monthly) // 5))
    train, test = monthly.iloc[:-test_count], monthly.iloc[-test_count:]
    history, previous, average = list(train), [], []
    for actual in test:
        previous.append(history[-1])
        average.append(float(np.mean(history[-3:])))
        history.append(float(actual))
    predictions = {"Previous month": np.array(previous), "3-month average": np.array(average)}
    if len(train) >= 12:
        predictions["Seasonal naive"] = monthly.shift(12).loc[test.index].to_numpy(float)
    scores = []
    for name, predicted in predictions.items():
        error = test.to_numpy() - predicted
        scores.append({"Baseline": name, "MAE": np.mean(np.abs(error)), "RMSE": np.sqrt(np.mean(error ** 2))})
    scores = pd.DataFrame(scores).sort_values("MAE").reset_index(drop=True)

    future_dates = pd.date_range(monthly.index[-1] + pd.offsets.MonthBegin(1), periods=6, freq="MS")
    future = {"Previous month": np.repeat(monthly.iloc[-1], 6)}
    history = list(monthly)
    moving_average = []
    for _ in range(6):
        value = float(np.mean(history[-3:]))
        moving_average.append(value)
        history.append(value)
    future["3-month average"] = moving_average
    if len(monthly) >= 12:
        future["Seasonal naive"] = monthly.iloc[-12:-6].to_numpy(float)
    return scores, pd.DataFrame(future, index=future_dates).rename_axis("Month")


def finish_change(message):
    """Show a message after the page reloads with the updated data."""
    st.session_state["flash"] = message
    st.rerun()


# Load the main dataset plus any records added from the app.
if not DATA_FILE.exists():
    st.error("cleaned_repairs_en.csv is missing. Put it in the same folder as app.py.")
    st.stop()
try:
    df = load_dataset(version(DATA_FILE), version(CATEGORY_FILE), version(USER_RECORDS))
except Exception as error:
    st.error(f"Could not load the repair dataset: {error}")
    st.stop()
if df.empty:
    st.error("The repair dataset contains no records.")
    st.stop()
user_count = int((df["source"] == USER_SOURCE).sum())

# Navigation and shared filters.
with st.sidebar:
    st.header("Menu")
    page = st.radio("Choose a page", ["Dashboard", "Repair Records", "Add / Upload Data", "Predict Repair Outcome",
                                      "Model Evaluation", "Fault Analysis", "Demand Trends"])
    st.header("Filters")
    categories = st.multiselect("Product category", choices(df, "product_category"), default=choices(df, "product_category"))
    brand_text = st.text_input("Brand contains", placeholder="Type a brand name")
    statuses = st.multiselect("Repair status", choices(df, "repair_status"), default=choices(df, "repair_status"))
    sources = st.multiselect("Data source", [ORIGINAL_SOURCE, USER_SOURCE], default=[ORIGINAL_SOURCE, USER_SOURCE])
    st.caption(f"{len(df):,} records loaded, including {user_count:,} user-added.")
    st.caption("SmartRepair AI project")

if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))

filtered = filter_data(df, categories, brand_text, statuses, sources)

if page == "Dashboard":
    st.header("Repair overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Repair records", f"{len(filtered):,}")
    c2.metric("User-added records", f"{(filtered['source'] == USER_SOURCE).sum():,}")
    c3.metric("Product categories", filtered["product_category"].nunique())
    c4.metric("Brands", filtered["brand"].nunique() if "brand" in filtered else "N/A")
    statuses_known = filtered["repair_status"].dropna().astype(str) if "repair_status" in filtered else pd.Series(dtype=str)
    statuses_known = statuses_known[statuses_known.str.lower() != "unknown"]
    fixed_rate = statuses_known.str.lower().eq("fixed").mean() * 100 if len(statuses_known) else np.nan
    c5.metric("Recorded fixed rate", f"{fixed_rate:.1f}%" if pd.notna(fixed_rate) else "N/A")

    left, right = st.columns(2)
    with left:
        st.subheader("Common appliance categories")
        st.bar_chart(filtered["product_category"].value_counts().head(12))
        if "brand" in filtered:
            st.subheader("Top brands")
            st.bar_chart(filtered["brand"].fillna("Unknown").value_counts().head(12))
    with right:
        st.subheader("Repair outcomes")
        if "repair_status" in filtered:
            st.bar_chart(filtered["repair_status"].fillna("Unknown").value_counts())
        if "event_date" in filtered:
            st.subheader("Repair records by year")
            yearly = filtered.dropna(subset=["event_date"]).set_index("event_date").resample("YS").size()
            if not yearly.empty:
                st.bar_chart(pd.Series(yearly.to_numpy(), index=yearly.index.year.astype(str)))

    if "repair_status" in filtered:
        st.subheader("Repair outcome by category")
        table = pd.crosstab(filtered["product_category"], filtered["repair_status"])
        st.bar_chart(table.reindex(filtered["product_category"].value_counts().head(12).index).fillna(0))
    if {"product_age", "repair_status"}.issubset(filtered.columns):
        age_data = filtered[filtered["product_age"].between(0, 100)].copy()
        if not age_data.empty:
            age_data["Age group"] = pd.cut(age_data["product_age"], [-0.01, 2, 5, 10, 15, 20, 100], labels=["0–2", "3–5", "6–10", "11–15", "16–20", "21+"])
            st.subheader("Outcome by product age")
            st.bar_chart(pd.crosstab(age_data["Age group"], age_data["repair_status"]))

    st.subheader("Quick business insights")
    if len(filtered):
        st.write(f"1. Most recorded category: **{filtered['product_category'].mode().iloc[0]}**. Review service capacity for this category.")
        if "brand" in filtered:
            st.write(f"2. Most recorded brand: **{filtered['brand'].fillna('Unknown').mode().iloc[0]}**. Review its common problems before planning stock.")
        if len(statuses_known):
            not_fixed = (~statuses_known.str.lower().eq("fixed")).mean() * 100
            st.write(f"3. **{not_fixed:.1f}%** of known recorded outcomes are not marked Fixed. Review diagnosis and customer communication.")
    st.caption("Community repair records may not represent demand for one commercial repair business.")

elif page == "Repair Records":
    st.header("Repair records")
    st.write(f"{len(filtered):,} records match your filters.")
    problem_col = "problem_en" if "problem_en" in filtered else "problem" if "problem" in filtered else None
    if problem_col:
        search = st.text_input("Search problem descriptions", placeholder="e.g. battery, screen, power")
        if search.strip():
            filtered = filtered[filtered[problem_col].fillna("").astype(str).str.contains(search.strip(), case=False, regex=False)]
    row_limit = st.selectbox("Rows to preview", [100, 500, 1000, 5000], index=1)
    preview = filtered.sort_values("source", ascending=False, kind="stable") if user_count else filtered
    st.dataframe(preview.head(row_limit), width="stretch", hide_index=True)
    st.download_button("Download filtered records", filtered.to_csv(index=False).encode("utf-8-sig"), "smartrepair_filtered_records.csv", "text/csv")

elif page == "Add / Upload Data":
    st.header("Add or upload repair records")
    st.write("New records are saved in `user_data/user_records.csv` and combined with the bundled dataset on every page. "
             "The bundled `cleaned_repairs_en.csv` is not changed.")
    add_tab, upload_tab, manage_tab = st.tabs(["Add one record", "Upload CSV file", f"Manage added records ({user_count:,})"])

    with add_tab:
        with st.form("add_record", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                category = st.selectbox("Product category", choices(df, "product_category"))
                brand = st.text_input("Brand", placeholder="e.g. Dell")
                countries = choices(df, "country")
                country = st.selectbox("Country", countries, index=countries.index("GBR") if "GBR" in countries else 0)
                provider = st.text_input("Data provider", value="SmartRepair user")
                status = st.selectbox("Repair status", choices(df, "repair_status"))
            with c2:
                event_date = st.date_input("Repair event date", value=pd.Timestamp.today())
                manufacture_year = st.number_input("Year of manufacture (optional)", 1900, pd.Timestamp.today().year, value=None, step=1)
                age = st.number_input("Product age in years (optional)", 0.0, 100.0, value=None, step=1.0)
                barrier = st.text_input("Repair barrier (optional)", placeholder="e.g. Spare parts not available")
                problem = st.text_area("Problem description (English)", placeholder="e.g. Screen flickers and turns off")
            submitted = st.form_submit_button("Add record", type="primary")
        if submitted:
            row = pd.DataFrame([{"product_category": category, "brand": brand or "Unknown", "country": country,
                                 "data_provider": provider, "repair_status": status, "event_date": event_date,
                                 "year_of_manufacture": manufacture_year, "product_age": age,
                                 "repair_barrier": barrier or None, "problem_en": problem or None}])
            records, notes = normalize_records(row, df)
            if records.empty:
                st.error(" ".join(notes) or "The record could not be added.")
            else:
                write_user_records(pd.concat([read_user_records(), records], ignore_index=True))
                finish_change(f"Added 1 record ({category}, {status}). All pages now include it.")

    with upload_tab:
        st.write("Upload a CSV with the same columns as the dataset. Required: **event_date**, **repair_status** and "
                 "**product_category** (or **product_category_id**). Put the English problem description in **problem_en** "
                 "(the original-language text can go in **problem**). Other columns are optional; extra columns are ignored.")
        template = pd.DataFrame([{"product_category": "Laptop", "brand": "Dell", "country": "GBR", "data_provider": "SmartRepair user",
                                  "year_of_manufacture": 2018, "product_age": 6, "repair_status": "Fixed", "repair_barrier": "",
                                  "event_date": pd.Timestamp.today().strftime("%Y-%m-%d"), "problem": "Akku lädt nicht",
                                  "problem_en": "Battery not charging"}])
        st.download_button("Download CSV template", template.to_csv(index=False).encode("utf-8-sig"), "smartrepair_template.csv", "text/csv")
        uploader_key = f"upload_{st.session_state.get('upload_count', 0)}"
        uploaded = st.file_uploader("Choose a CSV file", type="csv", key=uploader_key)
        if uploaded is not None:
            try:
                raw = pd.read_csv(uploaded, low_memory=False)
            except Exception as error:
                st.error(f"Could not read the CSV file: {error}")
            else:
                records, notes = normalize_records(raw, df)
                st.write(f"{len(raw):,} row(s) read, **{len(records):,}** ready to add.")
                for note in notes:
                    st.warning(note)
                if not records.empty:
                    st.dataframe(records.head(100), width="stretch", hide_index=True)
                    if st.button(f"Add {len(records):,} record(s) to the dataset", type="primary"):
                        write_user_records(pd.concat([read_user_records(), records], ignore_index=True))
                        st.session_state["upload_count"] = st.session_state.get("upload_count", 0) + 1
                        finish_change(f"Added {len(records):,} record(s) from {uploaded.name}. All pages now include them.")

    with manage_tab:
        added = read_user_records()
        if added.empty:
            st.info("No records have been added yet.")
        else:
            st.write("Tick records to remove them, or remove all of them to go back to the bundled dataset.")
            edited = st.data_editor(added.assign(Remove=False)[["Remove"] + RECORD_COLUMNS], disabled=RECORD_COLUMNS,
                                    hide_index=True, width="stretch", key=f"manage_{version(USER_RECORDS)}")
            c1, c2, c3 = st.columns(3)
            selected = edited.loc[edited["Remove"], "id"]
            if c1.button(f"Remove {len(selected):,} selected", disabled=selected.empty):
                write_user_records(added[~added["id"].isin(selected)])
                finish_change(f"Removed {len(selected):,} record(s).")
            if c2.button("Remove all added records"):
                write_user_records(added.iloc[0:0])
                finish_change("Removed all user-added records.")
            c3.download_button("Download added records", added.to_csv(index=False).encode("utf-8-sig"), "smartrepair_user_records.csv", "text/csv")

elif page == "Predict Repair Outcome":
    st.header("Predict repair outcome")
    st.write("Enter product details to estimate the recorded outcome from historical data.")
    model_file = artifact(MODEL_NAME)
    if not model_file.exists():
        st.error(f"{MODEL_NAME} is missing from the project folder.")
    else:
        if using_retrained_model():
            st.caption("Using the model retrained from the app (see Model Evaluation).")
        try:
            model = read_model(str(model_file), version(model_file))
            current_year = pd.Timestamp.today().year
            brands = df["brand"].fillna("Unknown").astype(str).value_counts().head(300).index.tolist() if "brand" in df else ["Unknown"]
            brand_options = ["Other / enter manually"] + brands
            with st.form("prediction_form"):
                c1, c2 = st.columns(2)
                with c1:
                    category = st.selectbox("Product category", choices(df, "product_category"))
                    brand_choice = st.selectbox("Brand", brand_options)
                    brand = st.text_input("Enter brand name") if brand_choice == "Other / enter manually" else brand_choice
                    country = st.selectbox("Country", choices(df, "country"))
                    provider = st.selectbox("Data provider", choices(df, "data_provider"))
                with c2:
                    age = st.number_input("Product age (years)", 0.0, 100.0, 5.0, 1.0)
                    manufacture_year = st.number_input("Year of manufacture", 1900, current_year, max(1900, current_year - 5))
                    repair_year = st.number_input("Expected repair year", 2000, current_year + 5, current_year)
                    repair_month = st.selectbox("Expected repair month", list(range(1, 13)), index=pd.Timestamp.today().month - 1)
                submitted = st.form_submit_button("Predict repair outcome")
            if submitted:
                rows = df[df["product_category"].astype(str) == category] if "product_category" in df else df.iloc[0:0]
                category_id = rows["product_category_id"].dropna().astype(str).mode().iloc[0] if not rows.empty and "product_category_id" in rows and not rows["product_category_id"].dropna().empty else "Unknown"
                values = {"product_age": age, "year_of_manufacture": manufacture_year, "event_year": repair_year, "event_month": repair_month, "product_category_id": category_id, "brand": (brand or "Unknown").upper(), "country": country, "data_provider": provider}
                features = list(getattr(model, "feature_names_in_", values.keys()))
                model_input = pd.DataFrame([{key: values.get(key, np.nan) for key in features}])
                for col in NUMERIC_FEATURES:
                    if col in model_input:
                        model_input[col] = pd.to_numeric(model_input[col], errors="coerce")
                for col in CATEGORICAL_FEATURES:
                    if col in model_input:
                        model_input[col] = model_input[col].fillna("Unknown").astype(str)
                prediction = model.predict(model_input)[0]
                st.success(f"Predicted recorded outcome: {prediction}")
                st.info(recommendation(prediction))
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(model_input)[0]
                    result = pd.DataFrame({"Outcome": model.classes_, "Estimated probability (%)": probabilities * 100}).sort_values("Estimated probability (%)", ascending=False)
                    st.subheader("Estimated probability by outcome")
                    st.bar_chart(result.set_index("Outcome"))
                    st.dataframe(result.round(2), width="stretch", hide_index=True)
                st.caption("This is an estimate, not a guarantee or a substitute for technician judgement.")
        except Exception as error:
            st.error(f"The prediction model could not be loaded or used: {error}")
            st.write("Check requirements.txt and make sure the model belongs to this project.")

elif page == "Model Evaluation":
    st.header("Member 2 — model evaluation")
    with st.container(border=True):
        st.subheader("Retrain with the current dataset")
        if using_retrained_model():
            st.write("The app is using a model **retrained from the app**. Its results are shown below.")
        else:
            st.write("The app is using the **bundled model**.")
        st.write(f"Retraining uses all {user_count:,} user-added records plus a stratified sample of the original data "
                 f"({TRAINING_SAMPLE:,} records in total). It usually takes under a minute.")
        c1, c2 = st.columns(2)
        if c1.button("Retrain model", type="primary"):
            try:
                with st.spinner("Training Logistic Regression..."):
                    retrain_model(df)
            except Exception as error:
                st.error(f"Retraining failed: {error}")
            else:
                finish_change("Model retrained. Predictions and evaluation now use the new model.")
        if c2.button("Restore bundled model", disabled=not using_retrained_model()):
            for name in MODEL_ARTIFACTS:
                (USER_DIR / name).unlink(missing_ok=True)
            finish_change("Restored the bundled model.")

    if MEMBER2_FILE.exists():
        st.subheader("Member 2: three-model comparison")
        results = pd.read_csv(MEMBER2_FILE)
        st.dataframe(results.round(4), width="stretch", hide_index=True)
        columns = [c for c in ["Accuracy", "Precision", "Recall", "F1 Score"] if c in results]
        if "Model" in results and columns:
            st.bar_chart(results.set_index("Model")[columns])
    if artifact(METRICS_NAME).exists():
        st.subheader("Retrained model evaluation" if using_retrained_model() else "Bundled model evaluation")
        metrics = pd.read_csv(artifact(METRICS_NAME))
        st.dataframe(metrics.round(4), width="stretch", hide_index=True)
        columns = [c for c in ["Accuracy", "Precision", "Recall", "F1 Score"] if c in metrics]
        if "Model" in metrics and columns:
            st.bar_chart(metrics.set_index("Model")[columns])
    if artifact(MATRIX_NAME).exists():
        st.subheader("Confusion matrix")
        st.dataframe(pd.read_csv(artifact(MATRIX_NAME), index_col=0), width="stretch")
    if artifact(REPORT_NAME).exists():
        with st.expander("Per-class precision, recall and F1-score"):
            st.dataframe(pd.read_csv(artifact(REPORT_NAME), index_col=0).round(3), width="stretch")
    if artifact(INFO_NAME).exists():
        with st.expander("Model training details"):
            st.json(json.loads(artifact(INFO_NAME).read_text(encoding="utf-8")))
    st.warning("These results describe this community dataset. Validate the model on a business's own records before operational use.")

elif page == "Fault Analysis":
    st.header("Member 3 — fault pattern analysis")
    if not FAULT_FILE.exists():
        st.info("Add member3_clustered_repairs.csv beside app.py to show fault results.")
    else:
        try:
            faults = load_fault_data(version(DATA_FILE), version(USER_RECORDS), version(FAULT_FILE))
            faults = filter_data(faults, categories, brand_text, statuses, sources)
            st.write(f"{len(faults):,} clustered records match your filters.")
            if user_count:
                st.caption("User-added records are placed in the nearest existing cluster using the Member 3 TF-IDF settings.")
            left, right = st.columns(2)
            with left:
                st.subheader("Records per cluster")
                st.bar_chart(faults["fault_cluster"].value_counts().sort_index())
            with right:
                st.subheader("Outcome by cluster (%)")
                st.bar_chart(pd.crosstab(faults["fault_cluster"], faults["repair_status"], normalize="index") * 100)
            st.subheader("Cluster outcome counts")
            st.dataframe(pd.crosstab(faults["fault_cluster"], faults["repair_status"]), width="stretch")
            st.subheader("Example problem descriptions")
            examples = faults.sort_values("source", ascending=False, kind="stable") if user_count else faults
            st.dataframe(examples[["fault_cluster", "problem_en", "source"]].dropna(subset=["problem_en"]).head(100), width="stretch", hide_index=True)
            with st.expander("Cluster summary"):
                summary = faults.groupby("fault_cluster").agg(record_count=("id", "size"), unique_brands=("brand", "nunique"))
                st.dataframe(summary, width="stretch")
            st.caption("Cluster numbers group similar descriptions; they are not confirmed technical fault labels.")
        except Exception as error:
            st.error(f"Could not load fault-analysis results: {error}")

elif page == "Demand Trends":
    st.header("Member 4 — repair demand and planning")
    if "event_date" not in filtered:
        st.info("The dataset does not contain event_date.")
    else:
        dated = filtered.dropna(subset=["event_date"])
        if dated.empty:
            st.info("No valid dates are available for the selected filters.")
        else:
            monthly = dated.set_index("event_date").resample("MS").size().asfreq("MS", fill_value=0)
            st.subheader("Monthly repair records")
            st.line_chart(monthly.rename("Repair records"))
            st.subheader("Common product categories")
            st.bar_chart(dated["product_category"].value_counts().head(12))
            problem_col = "problem_en" if "problem_en" in dated else "problem" if "problem" in dated else None
            if problem_col:
                st.subheader("Common reported problems")
                st.bar_chart(dated[problem_col].fillna("Unspecified").astype(str).value_counts().head(10))
            scores, predictions = forecast(monthly)
            if scores is None:
                st.info("At least eight monthly observations are needed to compare forecast baselines.")
            else:
                st.subheader("Baseline comparison (lower error is better)")
                st.dataframe(scores.round(2), width="stretch", hide_index=True)
                st.success(f"Lowest holdout MAE: {scores.iloc[0]['Baseline']}")
                st.subheader("Next six months: simple forecasts")
                st.line_chart(predictions)
                st.dataframe(predictions.round(1), width="stretch")
            st.warning("Forecasts estimate records in this dataset, not a business's actual demand or exact spare-parts quantities.")
