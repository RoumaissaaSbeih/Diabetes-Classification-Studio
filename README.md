# Diabetes Classification Studio

This project turns the classic Pima Indians diabetes classification workflow into a polished web app that covers the full last mile of data science:

- explore the data in a dashboard
- train and evaluate a deployment model
- save the trained artifact
- use that artifact live in the browser to score new patient profiles
- run everything locally with Docker or plain Python

## What is inside

- `train.py`: trains the model and exports artifacts to `artifacts/`
- `app.py`: launches the Streamlit dashboard
- `diabetes_dashboard/`: reusable code for loading data, training, inference, and UI
- `data/raw/diabetes.csv`: local copy of the dataset used by the app
- `Dockerfile` and `docker-compose.yml`: containerized runtime
- `.github/workflows/ci.yml`: GitHub Actions smoke test that installs dependencies and trains the model

## Modeling approach

The project is inspired by the Kaggle notebook by Shruti Iyyer and keeps the same spirit:

- clean zero values from medically invalid fields by treating them as missing
- split into train and test sets
- benchmark a few baseline classifiers
- tune a K-Nearest Neighbors classifier with `GridSearchCV`
- inspect the classification report, confusion matrix, ROC curve, and permutation importance
- serve the saved model inside a web app for new predictions

The final app uses the trained KNN artifact directly for live predictions.

## Local setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

Open `http://localhost:8501`.

## Run with Docker

```bash
docker compose up --build
```

Open `http://localhost:8501`.

## Dashboard sections

1. `Executive Dashboard`
   Data balance, feature relationships, and quick KPIs.
2. `Model Lab`
   Benchmark comparison, classification report, confusion matrix, ROC curve, and feature importance.
3. `Prediction Studio`
   Interactive patient form that sends values through the saved trained model and returns a risk probability.

## GitHub deployment flow

1. Create a new GitHub repository.
2. Add the remote:

```bash
git remote add origin https://github.com/<your-username>/diabetes-classification-studio.git
```

3. Push the code:

```bash
git add .
git commit -m "Build diabetes classification dashboard"
git branch -M main
git push -u origin main
```

The included GitHub Action will automatically validate that the model can train in CI.

## Notes

- This is an educational risk-classification demo, not a medical device.
- Zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` are treated as missing and median-imputed.
- The app auto-trains on first launch if artifacts are missing.

## Sources

- Kaggle notebook: [Step by Step Diabetes Classification](https://www.kaggle.com/code/shrutimechlearn/step-by-step-diabetes-classification)
- Referenced walkthrough confirming the KNN grid-search pattern and the tuned `n_neighbors` flow: [Building a k-Nearest-Neighbors (k-NN) Model with Scikit-learn](https://medium.com/data-science/building-a-k-nearest-neighbors-k-nn-model-with-scikit-learn-51209555453a)
