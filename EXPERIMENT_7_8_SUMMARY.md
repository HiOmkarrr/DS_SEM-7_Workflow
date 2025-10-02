# Experiments 7 & 8: CI/CD Pipeline & Dashboard Portfolio

## Experiment 7: CI/CD Pipeline with Open Source Tools ✅

### Objective
Automate testing, version checks, and deployment using GitHub Actions.

### Implementation
- **GitHub Actions Workflows:**
  - `ci.yml`: Lint, test, and build on push/PR to main
  - `test.yml`: Multi-Python version testing (3.9, 3.10, 3.11) with coverage
  - `docker-publish.yml`: Build and push Docker image to Docker Hub
  - `dvc.yml`: DVC pipeline for data versioning

- **Open Source Tools Used:**
  - GitHub Actions for CI/CD orchestration
  - Docker for containerization
  - DVC for data versioning
  - Ruff & Flake8 for linting
  - Pytest for testing

### Deliverables
- ✅ Workflow YAML files in `.github/workflows/`
- ✅ CI logs available in GitHub Actions tab
- ✅ Docker image automatically built and pushed to Docker Hub

### Key Features
- Multi-environment testing (Python 3.9-3.11)
- Automated Docker builds with multi-platform support
- DVC integration for data pipeline management
- Code coverage reporting
- Security scanning and dependency checks

---

## Experiment 8: Dashboard, Responsible AI Reporting & Final Portfolio ✅

### Objective
Build Streamlit dashboard, write Responsible AI report, and publish final repo.

### Implementation

#### Dashboard Features
- **Streamlit App** (`streamlit_app.py`):
  - Problem Statement page
  - Data Import/Scraping with Reddit PRAW integration
  - Comprehensive EDA with class balance, distributions, hypothesis testing
  - Model Selection (Linear Regression, Random Forest, XGBoost) with hyperparameter tuning
  - XAI & Fairness with SHAP, LIME, bias mitigation
  - Containerization with automated Docker build/run

#### API Service
- **FastAPI** (`api/main.py`):
  - Health check endpoint
  - Prediction endpoint with model validation
  - Automatic model loading and error handling
  - JSON-serializable responses

#### Responsible AI
- **Responsible_AI.md**: Comprehensive checklist covering:
  - Fairness evaluation across sensitive attributes
  - Privacy considerations for data handling
  - Consent requirements for data scraping
  - Transparency through SHAP/LIME explanations
  - Accountability through model versioning

#### Containerization
- **Dockerfile**: Multi-service container (API + Streamlit)
- **Docker Compose**: Local development setup
- **Automated deployment**: GitHub Actions builds and pushes to Docker Hub

### Deliverables
- ✅ Streamlit app: `streamlit run streamlit_app.py`
- ✅ API documentation: `http://localhost:8000/docs`
- ✅ Responsible_AI.md checklist
- ✅ Public GitHub repository with full CI/CD
- ✅ Docker Hub image: `[username]/ds-lab-app:latest`

### Key Features
- **End-to-end MLOps pipeline** from data ingestion to deployment
- **Explainable AI** with SHAP plots and LIME explanations
- **Fairness auditing** with bias mitigation strategies
- **Automated testing** and deployment
- **Data versioning** with DVC
- **Container orchestration** with Docker

---

## Repository Structure
```
├── .github/workflows/          # CI/CD workflows
├── api/                        # FastAPI service
├── prompt/                     # Data and lab files
├── tests/                      # Test suite
├── .dvc/                       # DVC configuration
├── streamlit_app.py           # Main dashboard
├── api/main.py                # API service
├── Dockerfile                 # Container definition
├── dvc.yaml                   # Data pipeline
├── Responsible_AI.md          # AI ethics checklist
└── requirements.txt           # Dependencies
```

## Usage Instructions

### Local Development
1. **Setup**: `pip install -r requirements.txt`
2. **Run Streamlit**: `streamlit run streamlit_app.py`
3. **Run API**: `uvicorn api.main:app --reload`
4. **Docker**: `docker build -t ds-lab-app . && docker run -p 8000:8000 -p 8501:8501 ds-lab-app`

### Production Deployment
- **Docker Hub**: Image automatically built on push to main
- **GitHub Actions**: Full CI/CD pipeline with testing and deployment
- **DVC**: Data versioning and pipeline management

## Conclusion

Both experiments successfully demonstrate modern MLOps practices:
- **Experiment 7** showcases robust CI/CD with automated testing, building, and deployment
- **Experiment 8** delivers a comprehensive dashboard with responsible AI practices and full-stack deployment

The implementation provides a production-ready template for data science projects with proper versioning, testing, and deployment automation.
