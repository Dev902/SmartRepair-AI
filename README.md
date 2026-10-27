# SmartRepair AI — Member 5 Integrated Application

A beginner-friendly Streamlit app that brings together the outputs from the five-member SmartRepair AI project using the Open Repair Alliance (ORA) repair-event dataset. You can add your own repair records from the browser, and every page updates to include them.

## What is included

- **Dashboard (Member 1):** category and brand frequencies, recorded outcomes, year trends, product-age groups, outcome-by-category charts, and data-derived business observations.
- **Repair Records:** shared category, brand-text, repair-status and data-source filters; problem-description search; CSV download.
- **Add / Upload Data:** add a single repair record with a form, upload a CSV of records, and review or remove the records you added.
- **Predict Repair Outcome (Member 2):** loads the prediction model, takes repair-time inputs, shows the predicted recorded status, estimated class probabilities and a cautious next-step recommendation.
- **Model Evaluation (Member 2):** the original three-model comparison, plus metrics for the model the app is using. A **Retrain model** button retrains it on the current dataset, including your records.
- **Fault Analysis (Member 3):** fault clusters, outcome distribution by cluster, example descriptions and a cluster summary.
- **Demand Trends (Member 4):** historical monthly record volumes, recurring categories/problems, and simple baseline forecasts compared on a recent holdout period.

## Adding your own data

Open **Add / Upload Data** in the sidebar.

- **Add one record:** fill in the form and click **Add record**.
- **Upload CSV file:** download the template, fill it in, and upload it. Required columns are `event_date`, `repair_status` and `product_category` (or `product_category_id`). Put the English problem description in `problem_en`; the original-language text can go in `problem`. If only `problem` is given, it is used as the English description. Other optional columns: `brand`, `country`, `data_provider`, `year_of_manufacture`, `product_age`, `repair_barrier`, `id`. The app shows a preview and any skipped rows before you confirm.
- **Manage added records:** tick records to remove them, remove all of them, or download them.

What happens to your records:

- They are saved in `user_data/user_records.csv`. The bundled `cleaned_repairs_en.csv` is never changed, so removing all added records restores the original results.
- Dashboard, Repair Records, Fault Analysis and Demand Trends include them straight away. Use the **Data source** filter to view only original or only user-added records.
- Category names and repair statuses are matched to existing values regardless of capitalisation; brand and country are stored in upper case like the original data. Rows whose `id` already exists in the dataset are skipped.
- `problem_en` (English) is the description used for search, common problems and fault clusters.
- In Fault Analysis, new records are placed in the nearest existing fault cluster using the same TF-IDF settings as Member 3's notebook. The first time this runs it takes about a minute; the result is cached in `user_data/`.
- The prediction model changes only when you click **Retrain model** on the Model Evaluation page. Retraining uses all user-added records plus a stratified sample of the original data (60,000 records in total) and saves the new model and its evaluation in `user_data/`. **Restore bundled model** switches back.

To start over completely, close the app and delete the `user_data` folder.

## Files

Keep these files in the same folder:

- `app.py`, `requirements.txt`
- `cleaned_repairs_en.csv` — the dataset; `problem_en` holds the English problem description
- `cleaned_categories (1).csv` — category names for `product_category_id`
- `best_repair_model.joblib` — bundled prediction model
- `model_comparison.csv`, `model_confusion_matrix.csv`, `model_classification_report.csv`, `model_training_info.json` — evaluation of the bundled model
- `member2_model_comparison.csv` — original three-model comparison from the team's ML notebook
- `member3_clustered_repairs.csv` — Member 3's fault clusters
- `smoke_test.py` — quick check of the bundled data and model

The team notebooks (`SmartRepair_AI_Fixed.ipynb`, `ML_Project.ipynb`, `fault_analysis(member_3).ipynb`, `Copy_of_Member4_Repair_Demand_Analysis.ipynb`) are kept for reference; they run in Google Colab. The raw ORA download is not bundled; get `OpenRepairData_v0.3_aggregate_202507.csv` from the Open Repair Alliance if you want to rerun the cleaning notebook.

## Run on Windows PowerShell

Open PowerShell in the folder containing `app.py`, then run:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If the `streamlit` command is not recognized, use the second command above exactly; it launches Streamlit through Python and does not require the executable to be on PATH. Open the local URL printed in the terminal, normally `http://localhost:8501`.

Optional: validate the bundled files before launching the UI:

```powershell
python smoke_test.py
```

## Model compatibility and evaluation notes

The model file originally supplied in the project was serialized with scikit-learn 1.6.1 and failed to load with newer scikit-learn because of an internal `ColumnTransformer` compatibility change. The bundled `best_repair_model.joblib` has therefore been retrained with **scikit-learn 1.8.0**, and the dependency is pinned to that version.

The bundled Logistic Regression model uses a fixed stratified sample of 60,000 records (48,000 train / 12,000 test). `member2_model_comparison.csv` preserves the original three-classifier scores from the Member 2 notebook; those scores are not directly comparable to the smaller compatibility model's metrics.

## Important interpretation limits

- ORA data describes community repair-event records. It may not represent a particular commercial repair chain's customers, costs or service volumes.
- `repair_status` is a historical recorded outcome. Predictions are estimates, not guarantees or a replacement for technician judgement.
- Fault-cluster numbers are mathematical groups, not verified technical fault names. Inspect descriptions before turning them into technician-training labels.
- Demand forecasts estimate counts of records in the dataset. The data does not establish exact spare-parts usage, inventory levels or quantities required by an individual business.
- Some product-age and manufacture-year fields are missing. The dashboard only plots available, valid values.
