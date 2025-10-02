import os
import re
import pandas as pd
import streamlit as st

DATA_DIR = os.path.join("prompt")
FINAL_DATA_PATH = os.path.join(DATA_DIR, "final_processed_zudio_data.csv")
RAW_DATA_PATH = os.path.join(DATA_DIR, "original_raw_data.csv")
PRAW_COLAB_FILE = os.path.join(DATA_DIR, "submitted_colab_files", "reddit_data_extraction.py")

st.set_page_config(page_title="DS Lab Experiments 7 & 8", layout="wide")

st.title("Data Science Lab: Experiments 7 & 8")

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Problem Statement",
        "Data Import / Scrape",
        "EDA",
        "Model Selection",
        "XAI & Fairness",
        "Containerization & API",
    ],
)

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read {path}: {exc}")
        return pd.DataFrame()


def _parse_praw_creds_from_file(colab_file_path: str):
    try:
        with open(colab_file_path, "r", encoding="utf-8") as f:
            content = f.read()
        cid = re.search(r"client_id=['\"]([^'\"]+)['\"]", content)
        csecret = re.search(r"client_secret=['\"]([^'\"]+)['\"]", content)
        uagent = re.search(r"user_agent=['\"]([^'\"]+)['\"]", content)
        return (
            cid.group(1) if cid else None,
            csecret.group(1) if csecret else None,
            uagent.group(1) if uagent else None,
        )
    except Exception:
        return (None, None, None)


def get_praw_credentials():
    client_id = os.getenv("PRAW_CLIENT_ID")
    client_secret = os.getenv("PRAW_CLIENT_SECRET")
    user_agent = os.getenv("PRAW_USER_AGENT", "ds-lab-app")
    if not client_id or not client_secret:
        file_id, file_secret, file_agent = _parse_praw_creds_from_file(PRAW_COLAB_FILE)
        if file_id and file_secret:
            st.info("Using PRAW credentials from reddit_data_extraction.py for demo.")
            client_id = client_id or file_id
            client_secret = client_secret or file_secret
            user_agent = user_agent or (file_agent or "ds-lab-app")
    return client_id, client_secret, user_agent


if page == "Problem Statement":
    st.subheader("Problem Statement")
    st.markdown(
        """
        This application demonstrates a full MLOps pipeline:
        1) Data ingestion (import and Reddit scraping), 2) Cleaning/Feature Engineering,
        3) EDA, 4) Modeling, 5) Explainability (SHAP, LIME) & Fairness (Fairlearn),
        6) Containerization & API, 7) CI/CD with GitHub Actions.
        
        Dataset used for Experiments 3–8: `prompt/final_processed_zudio_data.csv`.
        Experiment 2 is demonstrated on: `prompt/original_raw_data.csv`.
        """
    )
    st.success("Use the sidebar to navigate through the steps.")

elif page == "Data Import / Scrape":
    st.subheader("Data Import / Reddit Scrape")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Import Final Processed Dataset (Experiments 3–8)**")
        df_final = load_csv(FINAL_DATA_PATH)
        st.write(df_final.head())
        st.info(f"Rows: {len(df_final)} | Columns: {list(df_final.columns)}")
        if not df_final.empty and "product_engagement" in df_final.columns:
            st.markdown("**Default target: `product_engagement`**")
            st.caption("Reason: It captures continuous engagement (rating × log(1+count) plus trending/bestseller effects), enabling richer regression analysis and XAI/fairness on a meaningful business KPI.")
    with col2:
        st.markdown("**Import Original Raw Dataset (Experiment 2)**")
        df_raw = load_csv(RAW_DATA_PATH)
        st.write(df_raw.head())
        st.info(f"Rows: {len(df_raw)} | Columns: {list(df_raw.columns)}")
    st.divider()
    st.markdown("**Reddit Scraping (PRAW) - Optional Demo**")
    st.caption("Provide credentials via env vars OR embedded in submitted colab file. For production, use env vars.")
    subreddit = st.text_input("Subreddit", value="python")
    limit = st.slider("Number of posts", 1, 50, 5)
    if st.button("Demo Scrape"):
        try:
            import praw  # lazy import
            import prawcore
            client_id, client_secret, user_agent = get_praw_credentials()
            if not client_id or not client_secret:
                st.warning("Missing PRAW credentials; showing offline demo.")
                raise RuntimeError("Missing credentials")
            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
            )
            records = []
            try:
                # Try as a real subreddit name first
                sub = reddit.subreddit(subreddit)
                # Touch rules to force a permission check
                _ = sub.rules()
                records = [
                    {"title": s.title, "score": s.score, "num_comments": s.num_comments, "subreddit": str(s.subreddit)}
                    for s in sub.hot(limit=limit)
                ]
                st.success(f"Scraped from r/{subreddit}")
            except (prawcore.exceptions.Forbidden,
                    prawcore.exceptions.Redirect,
                    prawcore.exceptions.NotFound):
                # Fallback: treat input as keyword and search sitewide
                kw = subreddit
                sr_all = reddit.subreddit("all")
                records = [
                    {"title": s.title, "score": s.score, "num_comments": s.num_comments, "subreddit": str(s.subreddit)}
                    for s in sr_all.search(kw, limit=limit, sort="relevance")
                ]
                st.info(f"Subreddit r/{subreddit} not accessible. Searched sitewide for '{kw}' instead.")
            st.dataframe(pd.DataFrame.from_records(records))
        except Exception as exc:
            demo_df = pd.DataFrame(
                {"title": ["Demo post 1", "Demo post 2"], "score": [10, 5], "num_comments": [3, 1]}
            )
            st.info(f"Scrape demo due to: {exc}")
            st.dataframe(demo_df)

elif page == "EDA":
    st.subheader("Exploratory Data Analysis")
    df = load_csv(FINAL_DATA_PATH)
    if df.empty:
        st.error("Final processed dataset not found or empty.")
    else:
        st.write("Dataset preview:")
        st.dataframe(df.head(50))
        st.write("Basic statistics:")
        st.dataframe(df.describe(include="all").transpose())
        st.write("Column types:")
        st.json({c: str(t) for c, t in df.dtypes.items()})

        st.markdown("**Class balance analysis**")
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.ravel()
            # Category distribution
            if "category" in df.columns:
                counts = df["category"].value_counts().head(10)
                sns.barplot(x=counts.values, y=counts.index, ax=axes[0])
                axes[0].set_title("Top categories")
            # Availability
            if "availability" in df.columns:
                sns.countplot(data=df, x="availability", ax=axes[1])
                axes[1].set_title("Availability distribution")
                axes[1].tick_params(axis='x', rotation=30)
            # is_bestseller
            if "is_bestseller" in df.columns:
                sns.countplot(data=df, x="is_bestseller", ax=axes[2])
                axes[2].set_title("is_bestseller balance")
            # trending
            if "trending" in df.columns:
                sns.countplot(data=df, x="trending", ax=axes[3])
                axes[3].set_title("trending balance")
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as exc:
            st.info(f"Class balance skipped: {exc}")

        st.markdown("**Distributions and Rationale (inspired by Experiment 3)**")
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            st.caption("Histograms for numeric features (sampled if large)")
            sample_df = df.sample(min(1000, len(df)), random_state=42)
            cols = st.columns(2)
            for i, col in enumerate(numeric_cols[:6]):
                with cols[i % 2]:
                    st.bar_chart(sample_df[col].dropna())
        st.markdown("**Correlation heatmap (numeric columns)**")
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
            if numeric_cols:
                corr = df[numeric_cols].corr(numeric_only=True)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(corr, ax=ax, cmap="RdBu", center=0)
                st.pyplot(fig)
        except Exception as exc:
            st.info(f"Heatmap skipped: {exc}")

        st.markdown("**Distribution fitting for selected features**")
        try:
            import numpy as np
            from scipy import stats
            target_features = [c for c in ["price", "product_engagement", "rating_count", "rating"] if c in df.columns]
            if target_features:
                feature = st.selectbox("Select feature to fit distributions", options=target_features)
                values = df[feature].dropna()
                if len(values) > 20:
                    dists = {
                        "Normal": stats.norm,
                        "Exponential": stats.expon,
                        "Gamma": stats.gamma,
                        "Log-Normal": stats.lognorm,
                    }
                    aic_scores = {}
                    x = np.linspace(values.min(), values.max(), 200)
                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
                    axes = axes.flatten()
                    for i, (name, dist) in enumerate(dists.items()):
                        if name == "Exponential":
                            params = dist.fit(values, floc=0)
                        else:
                            params = dist.fit(values)
                        pdf = dist.pdf(x, *params)
                        loglik = np.sum(dist.logpdf(values, *params))
                        aic = 2 * len(params) - 2 * loglik
                        aic_scores[name] = aic
                        axes[i].hist(values, bins=30, density=True, alpha=0.6)
                        axes[i].plot(x, pdf, "r-")
                        axes[i].set_title(f"{name} (AIC={aic:.1f})")
                    plt.tight_layout()
                    st.pyplot(fig)
                    best = min(aic_scores, key=aic_scores.get)
                    st.success(f"Best fitting distribution for {feature}: {best}")
                    st.caption("Rationale: Compare plausible families and select by lowest AIC, following Experiment 3 methodology.")
                else:
                    st.info("Not enough data to fit distributions.")
        except Exception as exc:
            st.info(f"Distribution fitting skipped: {exc}")

        st.markdown("**Hypothesis testing: is there a significant difference in engagement?**")
        if "product_engagement" in df.columns:
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy.stats import ttest_ind
            # Explain the hypothesis once
            st.caption("H0: The mean product_engagement is the same across groups. H1: The means differ. We use Welch's t-test because group variances/sizes may differ.")
            for flag_col in ["trending", "is_bestseller"]:
                if flag_col in df.columns:
                    st.caption(f"Two-sample t-test: {flag_col} vs product_engagement")
                    try:
                        a = df.loc[df[flag_col].astype(str).str.lower().isin(["true", "1", "yes", "y"]) , "product_engagement"].dropna().astype(float)
                        b = df.loc[~df[flag_col].astype(str).str.lower().isin(["true", "1", "yes", "y"]) , "product_engagement"].dropna().astype(float)
                        if len(a) > 2 and len(b) > 2:
                            tstat, pval = ttest_ind(a, b, equal_var=False)
                            st.write(f"t={tstat:.3f}, p={pval:.4f} (Welch's t-test). Reason: comparing mean engagement between two independent groups.)")
                            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                            ax[0].boxplot([a, b], labels=[f"{flag_col}=True", f"{flag_col}=False"])
                            ax[0].set_title("Engagement distribution")
                            ax[1].hist(a, bins=30, alpha=0.6, label="True")
                            ax[1].hist(b, bins=30, alpha=0.6, label="False")
                            ax[1].legend(); ax[1].set_title("Histogram")
                            st.pyplot(fig)
                            # Interpretation helper
                            if pval < 0.05:
                                st.info("Interpretation: p<0.05 → reject H0. Evidence suggests mean engagement differs between groups.")
                            else:
                                st.info("Interpretation: p≥0.05 → fail to reject H0. No strong evidence of a mean difference.")
                        else:
                            st.info("Not enough samples in one of the groups.")
                    except Exception as exc:
                        st.info(f"t-test skipped: {exc}")

        # Chi-square test of independence between two categorical variables
        st.markdown("**Hypothesis testing: Chi-square test of independence (is_bestseller vs trending)**")
        if "is_bestseller" in df.columns and "trending" in df.columns:
            try:
                import seaborn as sns
                import matplotlib.pyplot as plt
                from scipy.stats import chi2_contingency
                st.caption("H0: Variables are independent (no association). H1: Variables are associated. We use Chi-square on the contingency table of counts.")
                # Normalize to boolean-like strings
                a_col = df["is_bestseller"].astype(str).str.lower().isin(["true", "1", "yes", "y"]).astype(int)
                b_col = df["trending"].astype(str).str.lower().isin(["true", "1", "yes", "y"]).astype(int)
                contingency = pd.crosstab(a_col, b_col)
                chi2, p, dof, expected = chi2_contingency(contingency)
                st.write("Contingency table (rows=is_bestseller, cols=trending):")
                st.dataframe(contingency)
                st.write(f"Chi-square={chi2:.3f}, dof={dof}, p-value={p:.4f}")
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(contingency, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
                ax.set_xlabel("trending (0/1)"); ax.set_ylabel("is_bestseller (0/1)")
                ax.set_title("Observed counts")
                st.pyplot(fig)
                if p < 0.05:
                    st.info("Interpretation: p<0.05 → reject H0. There is a significant association between is_bestseller and trending.")
                else:
                    st.info("Interpretation: p≥0.05 → fail to reject H0. No significant association detected.")
            except Exception as exc:
                st.info(f"Chi-square test skipped: {exc}")

elif page == "Model Selection":
    st.subheader("Model Selection & Training (Regression)")
    st.caption("Train Linear Regression, Random Forest, and XGBoost with preprocessing. Target must be numeric.")
    import numpy as np
    df = load_csv(FINAL_DATA_PATH)
    if df.empty:
        st.error("Dataset not available.")
    else:
        # Prefer default target if present
        target_candidates = [c for c in ["product_engagement"] if c in df.columns] or list(df.columns)
        target_col = st.selectbox("Select numeric target column", options=target_candidates)
        if target_col:
            y = df[target_col]
            if not pd.api.types.is_numeric_dtype(y):
                st.error("Selected target is not numeric. Please choose a numeric target for regression models.")
            else:
                X = df.drop(columns=[target_col])
                from sklearn.model_selection import train_test_split
                from sklearn.compose import ColumnTransformer
                from sklearn.preprocessing import OneHotEncoder, StandardScaler
                from sklearn.pipeline import Pipeline
                from sklearn.impute import SimpleImputer
                from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                import joblib

                numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
                categorical_cols = [c for c in X.columns if c not in numeric_cols]
                preprocessor = ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            numeric_cols,
                        ),
                        (
                            "cat",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="most_frequent")),
                                    # force dense output to avoid sparse errors in downstream estimators
                                    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                                ]
                            ),
                            categorical_cols,
                        ),
                    ]
                )

                model_options = ["LinearRegression", "RandomForestRegressor", "XGBRegressor"]
                selected_models = st.multiselect("Select models to train", options=model_options, default=model_options)
                st.markdown("**Hyperparameters**")
                with st.expander("Linear Regression"):
                    lr_fit_intercept = st.checkbox("fit_intercept", value=True)
                    lr_positive = st.checkbox("positive (non-negative coefficients)", value=False)
                with st.expander("Random Forest"):
                    rf_n_estimators = st.slider("n_estimators", 50, 500, 200, step=50)
                    rf_max_depth = st.slider("max_depth (0=none)", 0, 50, 0, step=1)
                    rf_min_samples_split = st.slider("min_samples_split", 2, 10, 2)
                with st.expander("XGBoost"):
                    xgb_n_estimators = st.slider("n_estimators", 100, 1000, 300, step=100)
                    xgb_learning_rate = st.select_slider("learning_rate", options=[0.01, 0.05, 0.1, 0.2], value=0.1)
                    xgb_max_depth = st.slider("max_depth", 2, 12, 6)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                def make_model(name: str):
                    if name == "LinearRegression":
                        from sklearn.linear_model import LinearRegression
                        model = Pipeline(steps=[("prep", preprocessor), ("model", LinearRegression(fit_intercept=lr_fit_intercept, positive=lr_positive))])
                        return model
                    if name == "RandomForestRegressor":
                        from sklearn.ensemble import RandomForestRegressor
                        model = Pipeline(steps=[("prep", preprocessor), ("model", RandomForestRegressor(n_estimators=rf_n_estimators, max_depth=None if rf_max_depth == 0 else rf_max_depth, min_samples_split=rf_min_samples_split, random_state=42))])
                        return model
                    if name == "XGBRegressor":
                        from xgboost import XGBRegressor
                        model = Pipeline(steps=[("prep", preprocessor), ("model", XGBRegressor(n_estimators=xgb_n_estimators, learning_rate=xgb_learning_rate, max_depth=xgb_max_depth, subsample=0.8, colsample_bytree=0.8, random_state=42, tree_method="hist", n_jobs=4))])
                        return model
                    raise ValueError("Unknown model")

                def evaluate(model):
                    preds = model.predict(X_test)
                    r2 = r2_score(y_test, preds)
                    mae = mean_absolute_error(y_test, preds)
                    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
                    return r2, mae, rmse

                colA, colB, colC = st.columns(3)
                results = {}

                if colA.button("Train Selected"):
                    for name in selected_models:
                        mdl = make_model(name)
                        mdl.fit(X_train, y_train)
                        r2, mae, rmse = evaluate(mdl)
                        results[name] = {"model": mdl, "R2": r2, "MAE": mae, "RMSE": rmse}
                    st.session_state["model_results"] = results

                if colB.button("Auto-run All (with quick hyperparam tuning)"):
                    from sklearn.model_selection import GridSearchCV
                    tuned_results = {}
                    from sklearn.linear_model import LinearRegression
                    lr_pipe = Pipeline(steps=[("prep", preprocessor), ("model", LinearRegression())])
                    lr_grid = {"model__fit_intercept": [True, False], "model__positive": [False, True]}
                    lr_cv = GridSearchCV(lr_pipe, lr_grid, cv=3, scoring="r2", n_jobs=-1, error_score="raise")
                    lr_cv.fit(X_train, y_train)
                    r2, mae, rmse = evaluate(lr_cv.best_estimator_)
                    tuned_results["LinearRegression"] = {"model": lr_cv.best_estimator_, "R2": r2, "MAE": mae, "RMSE": rmse, "best_params": lr_cv.best_params_}
                    from sklearn.ensemble import RandomForestRegressor
                    rf_pipe = Pipeline(steps=[("prep", preprocessor), ("model", RandomForestRegressor(random_state=42))])
                    rf_grid = {"model__n_estimators": [100, 300], "model__max_depth": [None, 10], "model__min_samples_split": [2, 5]}
                    rf_cv = GridSearchCV(rf_pipe, rf_grid, cv=3, scoring="r2", n_jobs=-1, error_score="raise")
                    rf_cv.fit(X_train, y_train)
                    r2, mae, rmse = evaluate(rf_cv.best_estimator_)
                    tuned_results["RandomForestRegressor"] = {"model": rf_cv.best_estimator_, "R2": r2, "MAE": mae, "RMSE": rmse, "best_params": rf_cv.best_params_}
                    from xgboost import XGBRegressor
                    xgb_pipe = Pipeline(steps=[("prep", preprocessor), ("model", XGBRegressor(random_state=42, tree_method="hist", n_jobs=4))])
                    xgb_grid = {"model__n_estimators": [200, 400], "model__max_depth": [4, 8], "model__learning_rate": [0.05, 0.1]}
                    xgb_cv = GridSearchCV(xgb_pipe, xgb_grid, cv=3, scoring="r2", n_jobs=-1, error_score="raise")
                    xgb_cv.fit(X_train, y_train)
                    r2, mae, rmse = evaluate(xgb_cv.best_estimator_)
                    tuned_results["XGBRegressor"] = {"model": xgb_cv.best_estimator_, "R2": r2, "MAE": mae, "RMSE": rmse, "best_params": xgb_cv.best_params_}
                    st.session_state["model_results"] = tuned_results

                if colC.button("Clear Results"):
                    st.session_state.pop("model_results", None)

                if "model_results" in st.session_state:
                    res = st.session_state["model_results"]
                    table = pd.DataFrame({k: {"R2": v["R2"], "MAE": v["MAE"], "RMSE": v["RMSE"]} for k, v in res.items()}).T
                    st.markdown("**Comparative Metrics**")
                    st.dataframe(table)
                    best_name = max(res.keys(), key=lambda n: (res[n]["R2"], -res[n]["RMSE"]))
                    st.success(f"Best model: {best_name} (R2={res[best_name]['R2']:.3f}, RMSE={res[best_name]['RMSE']:.3f})")
                    if "best_params" in res[best_name]:
                        st.markdown("**Best model hyperparameters**")
                        st.json(res[best_name]["best_params"]) 
                    
                    if st.button("Save Best Model & Config for CI/CD"):
                        import joblib
                        import json
                        
                        os.makedirs("models", exist_ok=True)
                        os.makedirs("config", exist_ok=True)
                        
                        # Save model bundle
                        bundle = {
                            "model": res[best_name]["model"],
                            "columns": list(X.columns),
                            "target": target_col,
                            "metrics": table.to_dict(),
                            "best_model": best_name,
                            "best_params": res[best_name].get("best_params"),
                        }
                        joblib.dump(bundle, "models/model.joblib")
                        
                        # Extract hyperparameters from best_params
                        best_params = res[best_name].get("best_params", {})
                        
                        # Create reproducible config
                        config = {
                            "model_type": best_name,
                            "target_column": target_col,
                            "feature_columns": list(X.columns),
                            "numeric_columns": numeric_cols,
                            "categorical_columns": categorical_cols,
                            "hyperparameters": {},
                            "preprocessing": {
                                "numeric_imputer": "median",
                                "numeric_scaler": "standard",
                                "categorical_imputer": "most_frequent",
                                "categorical_encoder": "onehot"
                            },
                            "train_test_split": {
                                "test_size": 0.2,
                                "random_state": 42
                            },
                            "metrics": {
                                "R2": float(res[best_name]["R2"]),
                                "MAE": float(res[best_name]["MAE"]),
                                "RMSE": float(res[best_name]["RMSE"])
                            },
                            "created_at": pd.Timestamp.now().isoformat()
                        }
                        
                        # Extract model-specific hyperparameters
                        if best_name == "LinearRegression":
                            config["hyperparameters"] = {
                                "fit_intercept": best_params.get("model__fit_intercept", True),
                                "positive": best_params.get("model__positive", False)
                            }
                        elif best_name == "RandomForestRegressor":
                            config["hyperparameters"] = {
                                "n_estimators": best_params.get("model__n_estimators", 200),
                                "max_depth": best_params.get("model__max_depth", None),
                                "min_samples_split": best_params.get("model__min_samples_split", 2),
                                "random_state": 42
                            }
                        elif best_name == "XGBRegressor":
                            config["hyperparameters"] = {
                                "n_estimators": best_params.get("model__n_estimators", 300),
                                "learning_rate": best_params.get("model__learning_rate", 0.1),
                                "max_depth": best_params.get("model__max_depth", 6),
                                "subsample": 0.8,
                                "colsample_bytree": 0.8,
                                "random_state": 42,
                                "tree_method": "hist",
                                "n_jobs": 4
                            }
                        
                        # Save config as JSON
                        config_path = "config/model_config.json"
                        with open(config_path, "w") as f:
                            json.dump(config, f, indent=2)
                        
                        st.success(f"✅ Saved model to models/model.joblib")
                        st.success(f"✅ Saved config to {config_path}")
                        st.info("Commit config/model_config.json to trigger CI/CD training with these exact parameters")
                        
                        # Show what to commit
                        st.code(f"""
git add config/model_config.json
git commit -m "Update model config: {best_name} with R2={res[best_name]['R2']:.3f}"
git push origin main
            """, language="bash")

elif page == "XAI & Fairness":
    st.subheader("Explainability & Fairness (Regression)")
    import joblib
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    model_path = Path("models/model.joblib")
    if not model_path.exists():
        st.warning("Train a model first in 'Model Selection'.")
    else:
        bundle = joblib.load(model_path)
        model = bundle["model"]
        columns = bundle["columns"]
        target = bundle.get("target")
        df = load_csv(FINAL_DATA_PATH)
        if df.empty:
            st.error("Dataset not available.")
        else:
            # Optional: derive delivery_days + speed_bucket like the notebook
            if "delivery_time" in df.columns and "delivery_days" not in df.columns:
                try:
                    df["delivery_days"] = df["delivery_time"].astype(str).str.extract(r"(\d+)").astype(float)
                    df["speed_bucket"] = pd.cut(df["delivery_days"], bins=[-np.inf, 3, 5, np.inf], labels=["Fast", "Standard", "Slow"])
                except Exception:
                    pass

            X = df[columns]
            y = df[target] if target in df.columns else None
            if y is None:
                st.error("Saved target column not found in dataset.")
            else:
                # Helper to get encoded matrix and names
                def get_encoded_and_names(X_in):
                    prep = model.named_steps["prep"]
                    X_enc = prep.transform(X_in)
                    if hasattr(X_enc, "toarray"):
                        X_enc = X_enc.toarray()
                    # names
                    try:
                        num_names = list(prep.transformers_[0][2])
                        ohe = prep.named_transformers_["cat"].named_steps["oh"]
                        cat_names = list(ohe.get_feature_names_out(prep.transformers_[1][2]))
                        names = num_names + cat_names
                    except Exception:
                        names = [f"f{i}" for i in range(X_enc.shape[1])]
                    return X_enc, names

                st.markdown("**SHAP: global and local explanations**")
                shap_ready = False
                bar_fig = None
                beeswarm_fig = None
                dep_fig = None
                try:
                    import shap
                    shap.sample = getattr(shap, "sample", shap)
                    # Prepare encoded small sample
                    sample_df = X.sample(min(200, len(X)), random_state=42)
                    X_enc, feature_names = get_encoded_and_names(sample_df)
                    # Try TreeExplainer on inner estimator; else permutation on pipeline predict
                    inner = getattr(model, "named_steps", {}).get("model", None)
                    try:
                        if inner is not None and inner.__class__.__name__ in ["XGBRegressor", "RandomForestRegressor"]:
                            explainer = shap.TreeExplainer(inner)
                            shap_values = explainer.shap_values(X_enc)
                        else:
                            raise RuntimeError("fallback")
                    except Exception:
                        masker = shap.maskers.Independent(X_enc)
                        explainer = shap.Explainer(lambda a: model.named_steps["model"].predict(a), masker, algorithm="permutation")
                        shap_values = explainer(X_enc).values
                    shap_ready = True
                    # Bar plot
                    try:
                        bar_fig = plt.figure(figsize=(8, 6))
                        shap.summary_plot(shap_values, X_enc, plot_type='bar', feature_names=feature_names, show=False, max_display=20)
                        st.pyplot(bar_fig)
                    except Exception:
                        pass
                    # Beeswarm
                    try:
                        beeswarm_fig = plt.figure(figsize=(8, 8))
                        shap.summary_plot(shap_values, X_enc, feature_names=feature_names, show=False, max_display=20)
                        st.pyplot(beeswarm_fig)
                    except Exception:
                        pass
                    # Dependence for top 4
                    try:
                        mean_abs = np.mean(np.abs(shap_values), axis=0)
                        top_idx = np.argsort(mean_abs)[-4:][::-1]
                        dep_fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                        axes = axes.ravel()
                        for i, idx in enumerate(top_idx):
                            axes[i].scatter(X_enc[:, idx], shap_values[:, idx], s=12, alpha=0.6)
                            axes[i].axhline(0, color='red', ls='--', alpha=0.5)
                            axes[i].set_xlabel(feature_names[idx])
                            axes[i].set_ylabel('SHAP value')
                        plt.tight_layout()
                        st.pyplot(dep_fig)
                    except Exception:
                        pass
                except Exception as exc:
                    st.info(f"SHAP unavailable: {exc}")
                    # Fallback: permutation_importance for global ranking
                    try:
                        from sklearn.inspection import permutation_importance
                        r = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=-1)
                        perm_df = pd.DataFrame({"feature": columns, "importance": r.importances_mean}).sort_values("importance", ascending=False).head(20)
                        st.dataframe(perm_df)
                    except Exception:
                        pass

                st.markdown("**LIME: local explanations**")
                try:
                    from lime.lime_tabular import LimeTabularExplainer
                    X_enc_full, feature_names = get_encoded_and_names(X)
                    inner = model.named_steps["model"]
                    explainer = LimeTabularExplainer(X_enc_full, feature_names=feature_names, mode='regression', discretize_continuous=True)
                    # pick three diverse samples: max, min, median target
                    y_order = y.reset_index(drop=True)
                    X_enc_order = X_enc_full
                    idx_max = int(y_order.idxmax()) if len(y_order) > 0 else 0
                    idx_min = int(y_order.idxmin()) if len(y_order) > 0 else 0
                    idx_med = int(np.argsort(y_order.values)[len(y_order)//2]) if len(y_order) > 0 else 0
                    selected = [("High target", idx_max), ("Low target", idx_min), ("Median target", idx_med)]
                    for title, irow in selected:
                        irow = max(0, min(irow, len(X_enc_order)-1))
                        exp = explainer.explain_instance(X_enc_order[irow], predict_fn=lambda a: inner.predict(a), num_features=10)
                        st.markdown(f"{title}")
                        st.json({k: float(v) for k, v in exp.as_list()})
                except Exception as exc:
                    st.info(f"LIME skipped: {exc}")

                # Add human-readable insights based on SHAP/global importances and errors
                try:
                    st.markdown("**Why are predictions behaving this way?**")
                    # Compute simple global correlations for intuition
                    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
                    insights = []
                    if target in df.columns and num_cols:
                        corr = df[num_cols + [target]].corr(numeric_only=True)[target].drop(target).sort_values(ascending=False)
                        top_pos = corr.head(3)
                        top_neg = corr.tail(3)
                        insights.append(f"Top features positively correlated with target: {', '.join([f'{k} ({v:.2f})' for k,v in top_pos.items()])}.")
                        insights.append(f"Top features negatively correlated with target: {', '.join([f'{k} ({v:.2f})' for k,v in top_neg.items()])}.")
                    # Residual behavior
                    preds_full = model.predict(X)
                    resid = preds_full - df[target]
                    over_rate = float((resid > 0).mean()) if len(resid) else 0.0
                    mean_abs_resid = float(resid.abs().mean()) if len(resid) else 0.0
                    insights.append(f"Overprediction rate: {over_rate*100:.1f}%; average absolute error: {mean_abs_resid:.3f}.")
                    # Short narrative
                    st.write(" ".join(insights))
                    st.caption("Correlations explain general directionality (not causation). SHAP plots show feature impact per prediction.")
                except Exception:
                    pass

                st.markdown("**Fairness audit and bias mitigation (regression)**")
                # Default sensitive feature like notebook: speed_bucket if present, else user-select
                default_opt = "speed_bucket" if "speed_bucket" in df.columns else "<select>"
                options = [default_opt] + [c for c in df.columns if c != default_opt]
                sensitive_col = st.selectbox("Sensitive feature column", options=options, index=0 if default_opt != "<select>" else 0)
                if sensitive_col != "<select>":
                    try:
                        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                        valid_mask = y.notna()
                        X_valid = X.loc[valid_mask]
                        y_valid = y.loc[valid_mask]
                        sens = df.loc[valid_mask, sensitive_col].astype(str)
                        preds = model.predict(X_valid)
                        groups_list = sorted(list(sens.unique()))
                        # Per-group metrics
                        rows = []
                        for g in groups_list:
                            mask = sens == g
                            if mask.sum() > 0:
                                group_true = y_valid[mask]
                                group_pred = preds[mask]
                                rows.append({
                                    "Group": str(g),
                                    "Count": int(mask.sum()),
                                    "Mean actual": float(group_true.mean()),
                                    "Mean pred": float(group_pred.mean()),
                                    "RMSE": float((mean_squared_error(group_true, group_pred)) ** 0.5),
                                    "MAE": float(mean_absolute_error(group_true, group_pred)),
                                    "R2": float(r2_score(group_true, group_pred) if len(group_true) > 1 else 0.0),
                                    "Bias": float(group_pred.mean() - group_true.mean()),
                                })
                        results_df = pd.DataFrame(rows)
                        display_df = results_df.copy()
                        for col in ["Mean actual", "Mean pred", "RMSE", "MAE", "R2", "Bias"]:
                            display_df[col] = display_df[col].astype(float).round(3)
                        st.dataframe(display_df)
                        # Mitigations
                        with st.expander("Bias mitigation & comparison"):
                            # Calibration
                            adjusted_cal = preds.copy()
                            cal_adjust = {}
                            for g in groups_list:
                                mask = sens == g
                                if mask.sum() > 0:
                                    bias = preds[mask].mean() - y_valid[mask].mean()
                                    adjusted_cal[mask] = preds[mask] - bias
                                    cal_adjust[g] = {"bias": float(bias), "adj": float(-bias)}
                            # Residual reweighting
                            def residual_reweight(y_true, y_pred, s, alpha=0.7):
                                out = y_pred.copy()
                                for g in groups_list:
                                    m = s == g
                                    if m.sum() > 0:
                                        corr = alpha * (y_true[m].mean() - y_pred[m].mean())
                                        out[m] = y_pred[m] + corr
                                return out
                            adjusted_rw = residual_reweight(y_valid, preds, sens)
                            # Compare bias disparity
                            def bias_disparity(y_true, y_pred, s):
                                biases = []
                                for g in groups_list:
                                    m = s == g
                                    if m.sum() > 0:
                                        biases.append(float(y_pred[m].mean() - y_true[m].mean()))
                                return float(max(biases) - min(biases)) if biases else 0.0
                            orig_disp = bias_disparity(y_valid, preds, sens)
                            cal_disp = bias_disparity(y_valid, adjusted_cal, sens)
                            rw_disp = bias_disparity(y_valid, adjusted_rw, sens)
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Bias disparity (orig)", f"{orig_disp:.3f}")
                            c2.metric("Bias disparity (calibrated)", f"{cal_disp:.3f}")
                            c3.metric("Bias disparity (reweighted)", f"{rw_disp:.3f}")
                    except Exception as exc:
                        st.info(f"Fairness analysis skipped: {exc}")

elif page == "Containerization & API":
    st.subheader("API Status & Containerization")
    st.markdown("This project includes a FastAPI service (`api/main.py`) and Dockerfile.")
    st.code("uvicorn api.main:app --reload --host 0.0.0.0 --port 8000")
    st.code("streamlit run streamlit_app.py --server.port 8501")
    st.info("CI/CD via GitHub Actions is configured in .github/workflows/ci.yml")

    st.markdown("**Automate container build & run (requires Docker Desktop)**")
    import subprocess, webbrowser, shutil
    docker_ok = shutil.which("docker") is not None
    if not docker_ok:
        st.warning("Docker not found on PATH. Install Docker Desktop and restart.")
    else:
        tag = st.text_input("Image tag", value="ds-lab-app")
        port_api = st.number_input("API port", 1, 65535, 8000)
        port_ui = st.number_input("UI port", 1, 65535, 8501)
        if st.button("Build & Run Container"):
            try:
                # Stop/remove existing container with same name
                subprocess.run(["docker", "rm", "-f", tag], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                build = subprocess.run(["docker", "build", "-t", tag, "."], check=True)
                run = subprocess.run([
                    "docker", "run", "-d", "--name", tag,
                    "-p", f"{port_api}:8000", "-p", f"{port_ui}:8501", tag
                ], check=True)
                st.success("Container running.")
                try:
                    webbrowser.open_new_tab(f"http://localhost:{port_api}/docs")
                except Exception:
                    pass
                st.markdown(f"Swagger: http://localhost:{port_api}/docs")
                st.markdown(f"Streamlit: http://localhost:{port_ui}")
            except subprocess.CalledProcessError as exc:
                st.error(f"Docker command failed: {exc}")
