# ❤️ AI-Based Cardiovascular Disease Prediction

An end-to-end Machine Learning project that predicts the likelihood of cardiovascular disease using clinical and lifestyle-related health parameters. This project implements and compares four classification algorithms and provides an interactive prediction dashboard built using Streamlit.

## 🚀 Project Overview

Cardiovascular diseases are among the leading causes of health-related complications worldwide. Early identification of potential risk patterns in health records can support awareness and further clinical evaluation.

This project uses machine learning classification algorithms to analyze patient health data and compare their predictive performance.

### Implemented Models

- Decision Tree Classifier
- Naive Bayes Classifier
- K-Nearest Neighbors (KNN)
- Artificial Neural Network (ANN – MLPClassifier)

## 🎯 Objectives

- Develop a machine learning-based cardiovascular disease prediction system.
- Compare four classification algorithms on a common dataset.
- Perform data preprocessing and feature engineering.
- Tune model hyperparameters using cross-validation.
- Evaluate models using standard classification metrics.
- Develop an interactive Streamlit application.
- Visualize model performance and feature importance.

## 📊 Dataset

- **Dataset:** Cardiovascular Disease Dataset
- **Records:** 70,000 patient records
- **File:** `cardio_train.csv`
- **Task:** Binary classification

The dataset contains medical and lifestyle-related attributes.

| Feature | Description |
|---|---|
| Age | Patient age |
| Gender | Patient gender |
| Height | Height in centimeters |
| Weight | Weight in kilograms |
| Ap_hi | Systolic blood pressure |
| Ap_lo | Diastolic blood pressure |
| Cholesterol | Cholesterol category |
| Gluc | Glucose category |
| Smoke | Smoking indicator |
| Alco | Alcohol consumption indicator |
| Active | Physical activity indicator |
| Cardio | Target variable |

<text>Target variable: `cardio` — indicates the presence or absence of cardiovascular disease in the dataset.</text>

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Web Application:** Streamlit
- **Model Serialization:** Joblib
- **Development Environment:** VS Code, Jupyter Notebook

## 🧠 Machine Learning Workflow

1. Data loading
2. Exploratory Data Analysis (EDA)
3. Data cleaning
4. Removal of duplicates and invalid records
5. Feature engineering
6. Train-test splitting
7. Feature scaling using pipelines
8. Hyperparameter tuning using GridSearchCV
9. Model training
10. Model evaluation
11. Model comparison
12. Model saving
13. Deployment using Streamlit

## 🧹 Data Preprocessing

The preprocessing pipeline includes:

- Removing duplicate records.
- Handling invalid medical values.
- Filtering implausible height and blood pressure values.
- Removing inconsistent records, such as systolic blood pressure less than or equal to diastolic pressure.
- Feature engineering.
- Scaling numerical features using Scikit-learn pipelines.

Feature scaling is performed within the model pipelines to prevent data leakage during training and cross-validation.

## 📈 Model Evaluation

The implemented classifiers are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The project also includes model comparison visualizations to analyze the performance of each classifier.

## 🏗️ Project Structure

```text
cvd_project/
│
├── app.py
├── train.py
├── src_common.py
├── cardio_ml_comparison.ipynb
├── requirements.txt
│
├── data/
│   └── cardio_train.csv
│
├── models/
│   ├── decision_tree.joblib
│   ├── naive_bayes.joblib
│   ├── knn.joblib
│   ├── ann.joblib
│   └── metrics.json
│
├── reports/
│   └── figures/
│
├── .vscode/
├── .streamlit/
│   └── config.toml
│
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

### 2. Navigate to the project directory

```bash
cd cvd_project
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows:**

```powershell
.\.venv\Scripts\Activate
```

**macOS/Linux:**

```bash
source .venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Train the models

```bash
python train.py
```

The training script generates trained models, evaluation metrics and report figures.

### 7. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

## 📓 Jupyter Notebook

Open `cardio_ml_comparison.ipynb` in Jupyter Notebook or VS Code.

Select the project's `.venv` Python kernel and run the notebook to explore:

- Exploratory Data Analysis
- Data preprocessing
- Model training
- Cross-validation
- Performance evaluation
- Model comparison
- Interpretation

## 🌐 Live Demo

**Live Application:** [CVD Risk Predictor](https://cvdproject-oyythrcee96zqer9eskfov.streamlit.app/)

## 🚀 Deployment

This project is deployed using Streamlit Community Cloud.

Deployment steps:

1. Push the project to GitHub.
2. Ensure `app.py`, `requirements.txt`, and the required trained model files are available in the repository.
3. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
4. Connect your GitHub repository.
5. Select `app.py` as the entry point.
6. Deploy the application.

## 🔮 Future Enhancements

- Advanced feature selection techniques.
- Comparison with additional machine learning models.
- Explainable AI using SHAP.
- Improved UI and dashboard analytics.
- Additional risk-factor visualization.
- Deployment with improved monitoring and testing.

## ⚠️ Disclaimer

This project is developed for educational and research purposes only. It is not a medical device and must not be used for medical diagnosis, treatment decisions, or emergency assessment.

Machine learning predictions are not a substitute for professional medical evaluation.

## 👩‍💻 Author

**Nitika Singh**

B.Tech Computer Science Engineering (AI & ML)

Interested in Machine Learning, Artificial Intelligence and Software Development.
