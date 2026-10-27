import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Electricity Consumption Dashboard", page_icon="⚡", layout="wide")

BASE = Path(__file__).parent
DATA_FILE = BASE / "preprocessed_electricity_data.csv"
RESULTS_FILE = BASE / "electricity_demand_forecasting_results.csv"
METRICS_FILE = BASE / "electricity_model_metrics.csv"
IMPORTANCE_FILE = BASE / "electricity_feature_importance.csv"
DATABASE_FILE = BASE / "electricity_datawarehouse.db"
MODEL_FILE = BASE / "electricity_model.pkl"

if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None
if "latest_input" not in st.session_state:
    st.session_state.latest_input = None


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def cached_csv(path, modified):
    return pd.read_csv(path)


def read_csv(path, warn=True):
    if not path.exists():
        if warn:
            st.warning(f"File not found: {path.name}")
        return None
    try:
        return cached_csv(str(path), path.stat().st_mtime)
    except Exception as exc:
        st.error(f"Could not read {path.name}: {exc}")
        return None


def find_column(columns, candidates):
    for candidate in candidates:
        for column in columns:
            if str(column).lower() == candidate.lower():
                return column
    return None


def quote_identifier(name):
    return '"' + str(name).replace('"', '""') + '"'


def show_df(df):
    st.dataframe(df, use_container_width=True)


def default_lag_value():
    """Realistic default for the lag inputs (median of saved actual demand)."""
    df = read_csv(RESULTS_FILE, warn=False)
    if df is None:
        return 1.0
    col = find_column(df.columns, ["Actual_Demand", "Actual", "Actual_Power"])
    if col is None:
        return 1.0
    value = pd.to_numeric(df[col], errors="coerce").median()
    return round(float(value), 3) if pd.notna(value) else 1.0


def load_results():
    """Clean saved forecasting results: DateTime, Actual Demand, Predicted Demand."""
    results = read_csv(RESULTS_FILE, warn=False)
    if results is None:
        return None
    actual_col = find_column(results.columns, ["Actual_Demand", "Actual", "Actual_Power"])
    predicted_col = find_column(results.columns, ["Predicted_Demand", "Predicted", "Predicted_Power"])
    date_col = find_column(results.columns, ["DateTime", "Date", "Timestamp"])
    if actual_col is None or predicted_col is None or date_col is None:
        st.error("Could not identify DateTime, actual and predicted demand columns.")
        st.write("Available columns:", list(results.columns))
        return None
    out = pd.DataFrame({
        "DateTime": pd.to_datetime(results[date_col], errors="coerce"),
        "Actual Demand": pd.to_numeric(results[actual_col], errors="coerce"),
        "Predicted Demand": pd.to_numeric(results[predicted_col], errors="coerce"),
    })
    out = out.dropna().sort_values("DateTime").reset_index(drop=True)
    return out if not out.empty else None


def parse_metrics(metrics):
    """Return {'MAE': .., 'RMSE': .., 'R2': ..} from the metrics CSV."""
    values = {}
    mc = find_column(metrics.columns, ["Metric", "Metrics", "Name"])
    vc = find_column(metrics.columns, ["Value", "Score", "Result"])
    if not (mc and vc):
        return values
    for _, row in metrics.iterrows():
        try:
            values[str(row[mc]).strip().upper()] = float(row[vc])
        except (ValueError, TypeError):
            pass
    return values


FEATURE_FALLBACK = {"Hour": 12, "Day": 15, "Month": 6, "Year": 2010, "Day_of_Week": 3, "Is_Weekend": 0}


def feature_baselines(features, lag_default):
    """Typical value of each feature (median of training data) used for the what-if analysis."""
    data = read_csv(DATA_FILE, warn=False)
    baselines = {}
    for feature in features:
        column = find_column(data.columns, [feature]) if data is not None else None
        if column is not None:
            value = pd.to_numeric(data[column], errors="coerce").median()
            if pd.notna(value):
                baselines[feature] = float(value)
                continue
        baselines[feature] = float(FEATURE_FALLBACK.get(feature, lag_default))
    return baselines


DEFAULT_LAG = default_lag_value()

# ----------------------------------------------------------------------------
# Load model
# ----------------------------------------------------------------------------
model = None
if MODEL_FILE.exists():
    try:
        model = load_model(str(MODEL_FILE))
    except Exception as exc:
        st.error(f"Could not load model {MODEL_FILE.name}: {exc}")
else:
    st.error(f"Model file not found: {MODEL_FILE.name}")

st.title("⚡ Electricity Consumption Pattern Analysis")
st.subheader("Demand Forecasting Using Data Warehousing and Data Mining")
st.write("Analyze electricity consumption, forecasting results, and OLAP operations.")
st.markdown("---")
section = st.sidebar.radio(
    "Navigate Dashboard",
    ["Home", "Consumption Analysis", "Random Forest Forecasting", "Data Warehouse", "OLAP Operations"],
)

# ----------------------------------------------------------------------------
# Shared prediction form
# ----------------------------------------------------------------------------
if model is not None:
    with st.expander("⚡ Input Demand Prediction", expanded=True):
        st.write("Change the values and click **Predict Demand**.")
        with st.form("electricity_prediction_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                hour = st.number_input("Hour", 0, 23, 12, step=1)
                day = st.number_input("Day", 1, 31, 15, step=1)
                month = st.number_input("Month", 1, 12, 6, step=1)
            with c2:
                year = st.number_input("Year", 2000, 2100, 2026, step=1)
                day_of_week = st.selectbox(
                    "Day of Week (Monday = 0, Sunday = 6)",
                    list(range(7)),
                    format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday",
                                           "Friday", "Saturday", "Sunday"][x],
                )
                is_weekend = st.selectbox("Is Weekend?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            with c3:
                lag_1 = st.number_input("Demand 1 hour ago (Lag_1)", min_value=0.0, value=DEFAULT_LAG, step=0.1, format="%.3f")
                lag_24 = st.number_input("Demand 24 hours ago (Lag_24)", min_value=0.0, value=DEFAULT_LAG, step=0.1, format="%.3f")
                lag_168 = st.number_input("Demand 168 hours ago (Lag_168)", min_value=0.0, value=DEFAULT_LAG, step=0.1, format="%.3f")
            predict_button = st.form_submit_button("Predict Demand")

        if predict_button:
            entered = pd.DataFrame([{
                "Hour": int(hour), "Day": int(day), "Month": int(month), "Year": int(year),
                "Day_of_Week": int(day_of_week), "Is_Weekend": int(is_weekend),
                "Lag_1": float(lag_1), "Lag_24": float(lag_24), "Lag_168": float(lag_168),
            }])
            try:
                if hasattr(model, "feature_names_in_"):
                    expected = list(model.feature_names_in_)
                    missing = [f for f in expected if f not in entered.columns]
                    if missing:
                        raise ValueError(f"Missing model feature(s): {missing}. Model expects: {expected}")
                    entered = entered[expected]
                st.session_state.latest_prediction = float(model.predict(entered)[0])
                st.session_state.latest_input = entered.copy()
                st.success("Prediction generated successfully!")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.info("Ensure the form feature names match the features used to train electricity_model.pkl.")

        if st.session_state.latest_prediction is not None:
            st.metric("Predicted Electricity Demand", f"{st.session_state.latest_prediction:,.2f}")
            st.caption("The latest submitted prediction updates every section on the forecasting page.")

# ----------------------------------------------------------------------------
# Home
# ----------------------------------------------------------------------------
if section == "Home":
    st.header("Welcome to the Electricity Dashboard")
    c1, c2, c3 = st.columns(3)
    c1.info("📊 Consumption Analysis")
    c1.write("Explore daily and hourly electricity usage patterns.")
    c2.info("🗄️ Data Warehouse")
    c2.write("Inspect the SQLite warehouse and OLAP operations.")
    c3.info("🤖 Demand Forecasting")
    c3.write("View saved forecasts, evaluation metrics, and feature importance.")
    st.markdown("---")
    st.subheader("Project Files")
    for p in [MODEL_FILE, DATA_FILE, RESULTS_FILE, METRICS_FILE, IMPORTANCE_FILE, DATABASE_FILE]:
        (st.success if p.exists() else st.error)(f"{'Found' if p.exists() else 'Missing'}: {p.name}")

# ----------------------------------------------------------------------------
# Consumption Analysis
# ----------------------------------------------------------------------------
elif section == "Consumption Analysis":
    st.header("📊 Electricity Consumption Analysis")
    df = read_csv(DATA_FILE)
    if df is not None:
        df = df.copy()
        date_col = find_column(df.columns, ["datetime", "date", "timestamp"])
        power_col = find_column(df.columns, ["global_active_power", "global active power"])
        if date_col is None or power_col is None:
            st.error("Required date or Global_active_power column was not found.")
            st.write("Available columns:", list(df.columns))
        else:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df[power_col] = pd.to_numeric(df[power_col], errors="coerce")
            df = df.dropna(subset=[date_col, power_col])
            if df.empty:
                st.warning("No valid consumption records were found.")
            else:
                a, b, c = st.columns(3)
                a.metric("Average Consumption", f"{df[power_col].mean():.3f} kW")
                b.metric("Maximum Consumption", f"{df[power_col].max():.3f} kW")
                c.metric("Minimum Consumption", f"{df[power_col].min():.3f} kW")

                st.subheader("Daily Electricity Consumption")
                daily = df.set_index(date_col)[power_col].resample("D").mean().dropna().reset_index()
                daily.columns = ["Date", "Average_Consumption"]
                if not daily.empty:
                    st.plotly_chart(
                        px.line(daily, x="Date", y="Average_Consumption",
                                title="Average Daily Electricity Consumption",
                                labels={"Average_Consumption": "Average Power (kW)"}),
                        use_container_width=True,
                    )

                st.subheader("Average Consumption by Hour")
                df["Hour"] = df[date_col].dt.hour
                hourly = df.groupby("Hour", as_index=False)[power_col].mean()
                hourly.columns = ["Hour", "Average_Consumption"]
                st.plotly_chart(
                    px.bar(hourly, x="Hour", y="Average_Consumption",
                           title="Average Electricity Consumption by Hour",
                           labels={"Hour": "Hour of Day", "Average_Consumption": "Average Power (kW)"}),
                    use_container_width=True,
                )
                st.caption("Average power in kW is different from total energy consumption in kWh.")

# ----------------------------------------------------------------------------
# Random Forest Forecasting  (every block below reacts to the prediction form)
# ----------------------------------------------------------------------------
elif section == "Random Forest Forecasting":
    st.header("🤖 Random Forest Demand Forecasting")

    prediction = st.session_state.latest_prediction
    latest_input = st.session_state.latest_input
    has_prediction = prediction is not None and latest_input is not None
    latest = latest_input.iloc[0] if has_prediction else None

    results = load_results()
    metrics = read_csv(METRICS_FILE, warn=False)
    metric_values = parse_metrics(metrics) if metrics is not None else {}

    # Average actual demand at the chosen hour (used in several blocks)
    hour_avg = None
    if has_prediction and results is not None:
        hour_avg = results.loc[results["DateTime"].dt.hour == int(latest["Hour"]), "Actual Demand"].mean()
        if pd.isna(hour_avg):
            hour_avg = None

    # Spread of the individual trees for this input (Random Forest uncertainty)
    tree_preds = None
    if has_prediction and hasattr(model, "estimators_"):
        try:
            X = latest_input.to_numpy(dtype=float)
            tree_preds = np.array([t.predict(X)[0] for t in model.estimators_])
        except Exception:
            tree_preds = None

    if not has_prediction:
        st.info("Enter values in the prediction form above and click **Predict Demand** "
                "to update all sections below.")

    # ---------------- 1. Model Evaluation Metrics ----------------
    st.subheader("Model Evaluation Metrics")
    if metrics is not None:
        show_df(metrics)
        a, b, c = st.columns(3)
        for label, aliases, container in [
            ("MAE", ["MAE", "MEAN ABSOLUTE ERROR"], a),
            ("RMSE", ["RMSE", "ROOT MEAN SQUARED ERROR"], b),
            ("R²", ["R2", "R²", "R-SQUARED", "R_SQUARED"], c),
        ]:
            for alias in aliases:
                if alias in metric_values:
                    container.metric(label, f"{metric_values[alias]:.4f}")
                    break
        st.caption("These are test-set scores of the trained model, so they stay the same for every input.")
    else:
        st.info("Model metrics CSV is not available.")

    if has_prediction:
        st.markdown("**Summary for your input**")
        s1, s2, s3 = st.columns(3)
        s1.metric("Predicted Demand", f"{prediction:.3f} kW")
        if tree_preds is not None:
            low, high = np.percentile(tree_preds, [10, 90])
            s2.metric("Likely range (10–90% of trees)", f"{low:.2f} – {high:.2f}")
        elif "RMSE" in metric_values:
            s2.metric("Likely range (± RMSE)",
                      f"{max(prediction - metric_values['RMSE'], 0):.2f} – {prediction + metric_values['RMSE']:.2f}")
        if hour_avg is not None:
            s3.metric(f"Avg actual demand at {int(latest['Hour']):02d}:00", f"{hour_avg:.3f} kW",
                      delta=f"{prediction - hour_avg:+.3f} vs your prediction")

    # ---------------- 2. Your Input chart ----------------
    st.markdown("---")
    st.subheader("Your Input: Predicted Demand")
    if has_prediction:
        bars = {
            "Lag_168 (1 week ago)": float(latest["Lag_168"]),
            "Lag_24 (1 day ago)": float(latest["Lag_24"]),
            "Lag_1 (1 hour ago)": float(latest["Lag_1"]),
            "Predicted Demand": float(prediction),
        }
        if hour_avg is not None:
            bars["Avg actual (same hour)"] = float(hour_avg)
        input_chart_df = pd.DataFrame({"Type": list(bars.keys()), "Demand": list(bars.values())})
        fig_input = px.bar(input_chart_df, x="Type", y="Demand", color="Type", text="Demand",
                           title="Your Inputs vs Predicted Demand")
        fig_input.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_input.update_layout(showlegend=False)
        st.plotly_chart(fig_input, use_container_width=True, key="user_input_prediction_chart")
    else:
        st.info("Submit the prediction form to see this chart.")

    # ---------------- 3. Actual vs Predicted ----------------
    st.markdown("---")
    st.subheader("Actual vs Predicted Electricity Demand")
    if results is None:
        st.info(f"Forecasting results file not available: {RESULTS_FILE.name}")
    else:
        # 3a. Average by hour of day + your prediction
        profile = (results.assign(Hour=results["DateTime"].dt.hour)
                   .groupby("Hour")[["Actual Demand", "Predicted Demand"]].mean().reset_index())
        profile_long = profile.melt(id_vars="Hour", var_name="Series", value_name="Demand")
        fig_profile = px.line(profile_long, x="Hour", y="Demand", color="Series", markers=True,
                              title="Average Actual vs Predicted Demand by Hour of Day",
                              labels={"Hour": "Hour of Day", "Demand": "Electricity Demand (kW)"})
        if has_prediction:
            fig_profile.add_trace(go.Scatter(
                x=[int(latest["Hour"])], y=[float(prediction)], mode="markers+text",
                name="Your Prediction", text=[f"{prediction:.2f}"], textposition="top center",
                marker=dict(size=16, symbol="star", color="red"),
            ))
        st.plotly_chart(fig_profile, use_container_width=True, key="hourly_profile_chart")

        # 3b. Recent saved records (zoomed so the lines are readable)
        max_n = min(len(results), 2000)
        if max_n > 24:
            n_records = st.slider("Number of recent records to display", 24, max_n, min(168, max_n), step=24)
        else:
            n_records = max_n
        recent = results.tail(n_records)
        recent_long = recent.melt(id_vars="DateTime", value_vars=["Actual Demand", "Predicted Demand"],
                                  var_name="Series", value_name="Demand")
        fig_forecast = px.line(recent_long, x="DateTime", y="Demand", color="Series",
                               title="Actual vs Predicted Electricity Demand (saved test results)",
                               labels={"DateTime": "Date / Time", "Demand": "Electricity Demand (kW)"})
        st.plotly_chart(fig_forecast, use_container_width=True, key="actual_vs_predicted_chart")

    # ---------------- 4. Forecasting Results Preview ----------------
    st.markdown("---")
    st.subheader("Forecasting Results Preview")
    if results is None:
        st.info("No saved forecasting results to preview.")
    elif has_prediction:
        h, d = int(latest["Hour"]), int(latest["Day_of_Week"])
        similar = results[(results["DateTime"].dt.hour == h) & (results["DateTime"].dt.dayofweek == d)]
        note = "same hour and same weekday"
        if similar.empty:
            similar = results[results["DateTime"].dt.hour == h]
            note = "same hour"
        user_row = pd.DataFrame([{
            "DateTime": f"{int(latest['Year'])}-{int(latest['Month']):02d}-{int(latest['Day']):02d} {h:02d}:00 (your input)",
            "Actual Demand": np.nan,
            "Predicted Demand": float(prediction),
        }])
        history = similar.head(10).copy()
        history["DateTime"] = history["DateTime"].astype(str)
        show_df(pd.concat([user_row, history], ignore_index=True))
        st.caption(f"First row is your prediction; the rows below are saved results with the {note}.")
    else:
        show_df(results.head(10))
        st.caption("Submit the prediction form to compare your input with similar historical hours.")

    # ---------------- 5. Feature Importance ----------------
    st.markdown("---")
    st.subheader("Feature Importance")
    importance = read_csv(IMPORTANCE_FILE, warn=False)
    if importance is not None:
        feature_col = find_column(importance.columns, ["Feature", "Features", "Variable"])
        importance_col = find_column(importance.columns, ["Importance", "Feature_Importance", "Importance_Score"])
        if feature_col and importance_col:
            importance = importance.copy()
            importance[importance_col] = pd.to_numeric(importance[importance_col], errors="coerce")
            importance = importance.dropna(subset=[importance_col]).sort_values(importance_col, ascending=False)
            fig_imp = px.bar(importance, x=importance_col, y=feature_col, orientation="h",
                             title="Random Forest Feature Importance (global, from training)",
                             labels={feature_col: "Feature", importance_col: "Importance"})
            fig_imp.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_imp, use_container_width=True)
            show_df(importance)
        else:
            st.error("Could not identify feature and importance columns.")
            st.write("Available columns:", list(importance.columns))
    else:
        st.info("Feature importance CSV is not available.")

    if has_prediction:
        st.markdown("**Impact of each input on your prediction**")
        try:
            features = list(latest_input.columns)
            baselines = feature_baselines(features, DEFAULT_LAG)
            rows = []
            for feature in features:
                alt = latest_input.astype(float).copy()
                alt[feature] = baselines[feature]
                rows.append({
                    "Feature": feature,
                    "Your value": float(latest_input.iloc[0][feature]),
                    "Typical value": baselines[feature],
                    "Impact on prediction": float(prediction) - float(model.predict(alt)[0]),
                })
            impact = pd.DataFrame(rows)
            impact["Direction"] = np.where(impact["Impact on prediction"] >= 0, "Increases demand", "Decreases demand")
            impact = impact.reindex(impact["Impact on prediction"].abs().sort_values(ascending=False).index)
            fig_impact = px.bar(impact, x="Impact on prediction", y="Feature", orientation="h", color="Direction",
                                color_discrete_map={"Increases demand": "#d62728", "Decreases demand": "#1f77b4"},
                                title="How each input moves the prediction compared with a typical value")
            fig_impact.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_impact, use_container_width=True, key="input_impact_chart")
            show_df(impact.drop(columns=["Direction"]))
            st.caption("Global importance above is fixed after training. This chart is recalculated for every "
                       "input: each feature is replaced by its typical value and the change in prediction is shown.")
        except Exception as exc:
            st.warning(f"Could not compute input impact: {exc}")

# ----------------------------------------------------------------------------
# Data Warehouse
# ----------------------------------------------------------------------------
elif section == "Data Warehouse":
    st.header("🗄️ Electricity Data Warehouse")
    if not DATABASE_FILE.exists():
        st.error("The SQLite database file was not found.")
    else:
        try:
            with sqlite3.connect(DATABASE_FILE) as conn:
                tables = pd.read_sql_query(
                    "SELECT name AS Table_Name FROM sqlite_master WHERE type='table' ORDER BY name", conn)
                st.subheader("Database Tables")
                show_df(tables)
                for table_name in tables["Table_Name"]:
                    with st.expander(f"Table: {table_name}"):
                        cols = pd.read_sql_query(f"PRAGMA table_info({quote_identifier(table_name)})", conn)
                        st.write("Columns")
                        show_df(cols[["name", "type"]])
                        sample = pd.read_sql_query(f"SELECT * FROM {quote_identifier(table_name)} LIMIT 5", conn)
                        st.write("Sample Records")
                        show_df(sample)
        except Exception as exc:
            st.error(f"Database error: {exc}")

# ----------------------------------------------------------------------------
# OLAP Operations
# ----------------------------------------------------------------------------
elif section == "OLAP Operations":
    st.header("📊 OLAP Operations")
    st.write("Explore the data warehouse using Roll-up, Drill-down, Slice, Dice, and Pivot.")
    if not DATABASE_FILE.exists():
        st.error("The SQLite database file was not found.")
    else:
        try:
            with sqlite3.connect(DATABASE_FILE) as conn:
                tables = pd.read_sql_query(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn)["name"].tolist()
                fact_table = next((t for t in tables if t.lower() == "fact_electricity"), None)
                date_table = next((t for t in tables if t.lower() == "dim_date"), None)
                if not fact_table or not date_table:
                    st.error("Required tables Fact_Electricity and Dim_Date were not found.")
                    st.write("Tables found:", tables)
                else:
                    fact_cols = pd.read_sql_query(f"PRAGMA table_info({quote_identifier(fact_table)})", conn)["name"].tolist()
                    date_cols = pd.read_sql_query(f"PRAGMA table_info({quote_identifier(date_table)})", conn)["name"].tolist()
                    power_col = find_column(fact_cols, ["global_active_power", "consumption", "power"])
                    fk_col = find_column(fact_cols, ["date_key", "date_id"])
                    dk_col = find_column(date_cols, ["date_key", "date_id"])
                    year_col = find_column(date_cols, ["year"])
                    month_col = find_column(date_cols, ["month"])
                    day_col = find_column(date_cols, ["day"])

                    if not all([power_col, fk_col, dk_col, year_col, month_col]):
                        st.error("The database schema differs from the expected structure.")
                        st.write("Fact table columns:", fact_cols)
                        st.write("Date table columns:", date_cols)
                    else:
                        f, d = quote_identifier(fact_table), quote_identifier(date_table)
                        p, fk, dk = quote_identifier(power_col), quote_identifier(fk_col), quote_identifier(dk_col)
                        y, m = quote_identifier(year_col), quote_identifier(month_col)
                        day = quote_identifier(day_col) if day_col else None
                        join = f" FROM {f} AS f JOIN {d} AS d ON f.{fk}=d.{dk} "
                        tabs = st.tabs(["Roll-up", "Drill-down", "Slice", "Dice", "Pivot"])

                        # Roll-up
                        with tabs[0]:
                            st.subheader("Roll-up: Monthly Average Consumption")
                            q = (f"SELECT d.{y} AS Year,d.{m} AS Month,AVG(f.{p}) AS Average_Consumption" + join +
                                 f"GROUP BY d.{y},d.{m} ORDER BY d.{y},d.{m}")
                            df = pd.read_sql_query(q, conn)
                            if not df.empty:
                                df["Year_Month"] = df["Year"].astype(str) + "-" + df["Month"].astype(int).astype(str).str.zfill(2)
                                show_df(df)
                                st.plotly_chart(px.bar(df, x="Year_Month", y="Average_Consumption",
                                                       title="Monthly Average Electricity Consumption"),
                                                use_container_width=True)
                            else:
                                st.warning("No monthly consumption data is available.")

                        # Drill-down
                        with tabs[1]:
                            st.subheader("Drill-down: Year to Month to Day")
                            years = pd.read_sql_query(f"SELECT DISTINCT {y} AS Year FROM {d} ORDER BY Year", conn)["Year"].dropna().tolist()
                            if years:
                                sy = st.selectbox("Select Year", years, key="drill_year")
                                months = pd.read_sql_query(f"SELECT DISTINCT {m} AS Month FROM {d} WHERE {y}=? ORDER BY Month",
                                                           conn, params=[sy])["Month"].dropna().tolist()
                                if months:
                                    sm = st.selectbox("Select Month", months, key="drill_month")
                                    if day:
                                        df = pd.read_sql_query(
                                            f"SELECT d.{y} AS Year,d.{m} AS Month,d.{day} AS Day,AVG(f.{p}) AS Average_Consumption" + join +
                                            f"WHERE d.{y}=? AND d.{m}=? GROUP BY d.{y},d.{m},d.{day} ORDER BY d.{day}",
                                            conn, params=[sy, sm])
                                        show_df(df)
                                        if not df.empty:
                                            st.plotly_chart(px.bar(df, x="Day", y="Average_Consumption",
                                                                   title="Daily Average Consumption"),
                                                            use_container_width=True)
                                    else:
                                        st.warning("The date table has no Day column.")
                                else:
                                    st.info("No months found for this year.")
                            else:
                                st.info("No years are available.")

                        # Slice
                        with tabs[2]:
                            st.subheader("Slice: Consumption for One Year")
                            years = pd.read_sql_query(f"SELECT DISTINCT {y} AS Year FROM {d} ORDER BY Year", conn)["Year"].dropna().tolist()
                            if years:
                                sy = st.selectbox("Select year for Slice", years, key="slice_year")
                                df = pd.read_sql_query(
                                    f"SELECT d.{m} AS Month,AVG(f.{p}) AS Average_Consumption" + join +
                                    f"WHERE d.{y}=? GROUP BY d.{m} ORDER BY d.{m}", conn, params=[sy])
                                show_df(df)
                                if not df.empty:
                                    st.plotly_chart(px.bar(df, x="Month", y="Average_Consumption",
                                                           title=f"Monthly Consumption in {sy}"),
                                                    use_container_width=True)
                            else:
                                st.info("No years are available.")

                        # Dice
                        with tabs[3]:
                            st.subheader("Dice: Filter by Year and Month")
                            years = pd.read_sql_query(f"SELECT DISTINCT {y} AS Year FROM {d} ORDER BY Year", conn)["Year"].dropna().tolist()
                            if years:
                                sy = st.selectbox("Select year for Dice", years, key="dice_year")
                                months = pd.read_sql_query(f"SELECT DISTINCT {m} AS Month FROM {d} WHERE {y}=? ORDER BY Month",
                                                           conn, params=[sy])["Month"].dropna().tolist()
                                if months:
                                    sm = st.selectbox("Select month for Dice", months, key="dice_month")
                                    select_cols = f"d.{y} AS Year,d.{m} AS Month," + (f"d.{day} AS Day," if day else "")
                                    group_cols = f"d.{y},d.{m}" + (f",d.{day}" if day else "")
                                    order = f" ORDER BY d.{day}" if day else ""
                                    df = pd.read_sql_query(
                                        f"SELECT {select_cols} AVG(f.{p}) AS Average_Consumption" + join +
                                        f"WHERE d.{y}=? AND d.{m}=? GROUP BY {group_cols}{order}",
                                        conn, params=[sy, sm])
                                    show_df(df)
                                    st.caption("Results are filtered by both year and month.")
                                else:
                                    st.info("No months found for this year.")
                            else:
                                st.info("No years are available.")

                        # Pivot
                        with tabs[4]:
                            st.subheader("Pivot: Monthly Consumption by Year")
                            df = pd.read_sql_query(
                                f"SELECT d.{y} AS Year,d.{m} AS Month,AVG(f.{p}) AS Average_Consumption" + join +
                                f"GROUP BY d.{y},d.{m} ORDER BY d.{y},d.{m}", conn)
                            if not df.empty:
                                pivot = df.pivot(index="Month", columns="Year", values="Average_Consumption")
                                st.write("Rows represent months; columns represent years.")
                                show_df(pivot.reset_index())
                                df["Year"] = df["Year"].astype(str)
                                st.plotly_chart(px.line(df, x="Month", y="Average_Consumption", color="Year",
                                                        markers=True, title="Monthly Average Consumption Across Years"),
                                                use_container_width=True)
                            else:
                                st.warning("No data is available for the pivot.")
        except Exception as exc:
            st.error("An error occurred while loading the OLAP operations.")
            st.exception(exc)

st.markdown("---")
st.caption("Electricity Consumption Pattern Analysis and Demand Forecasting Using Data Warehousing and Data Mining")
