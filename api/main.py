# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, Optional
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Zudio ML API")

MODEL_PATH = Path("models/model.joblib")
_model = None
_columns = None
_target = None

class PredictRequest(BaseModel):
	features: Dict[str, Any]

@app.on_event("startup")
def load_model_on_startup():
	global _model, _columns, _target
	if MODEL_PATH.exists():
		try:
			bundle = joblib.load(MODEL_PATH)
			_model = bundle.get("model")
			_columns = bundle.get("columns")
			_target = bundle.get("target")
		except Exception:
			_model = None
			_columns = None
			_target = None

@app.get("/")
def root():
	return {"status": "ok", "model_loaded": _model is not None, "columns": _columns}

@app.get("/health")
def health_check():
	if _model is None or _columns is None:
		return {"status": "degraded", "detail": "Model not loaded. Train and save via Streamlit."}
	try:
		df = pd.DataFrame([{c: None for c in _columns}])
		_ = _model.predict(df)
		return {"status": "healthy", "model_loaded": True, "num_features": len(_columns)}
	except Exception as exc:
		return {"status": "unhealthy", "error": str(exc)}

@app.post("/predict")
def predict(req: PredictRequest):
	if _model is None or _columns is None:
		return {"detail": "Model not found. Train and save best model in Streamlit first."}
	# Align input to training columns; fill missing with None
	row = {c: req.features.get(c, None) for c in _columns}
	df = pd.DataFrame([row], columns=_columns)
	try:
		pred = _model.predict(df)
		value: Optional[Any] = pred[0] if len(pred) else None
		# Convert numpy types to Python scalars for JSON
		if isinstance(value, (np.generic,)):
			value = np.asarray(value).item()
		elif hasattr(value, "item"):
			try:
				value = value.item()
			except Exception:
				pass
		result = {"prediction": value}
		# Optionally include proba if the model supports it (classification)
		if hasattr(_model, "predict_proba"):
			try:
				proba_arr = _model.predict_proba(df)[0]
				result["proba"] = {str(i): float(p) for i, p in enumerate(proba_arr)}
			except Exception:
				pass
		return result
	except Exception as exc:
		return {"error": str(exc), "echo": row}
