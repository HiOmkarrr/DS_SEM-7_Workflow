# Responsible AI Checklist

- Fairness: Identify sensitive attributes; evaluate parity metrics (selection rate) with Fairlearn.
- Transparency: Provide SHAP/LIME explanations for model predictions.
- Privacy: Avoid storing PII; restrict data exposure in API responses.
- Consent: Ensure data scraping complies with subreddit rules and Reddit API terms.
- Robustness: Validate inputs; handle missing values; monitor for drift.
- Accountability: Version models and data; document changes in README/CI logs.

## How to Use in This Project
- Use Streamlit page "XAI & Fairness" to compute explanations and parity metrics.
- Provide `sensitive_col` to compare groups.
- Retrain if significant disparities are observed.
