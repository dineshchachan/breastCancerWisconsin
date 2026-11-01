# Import the libraries needed for the app.
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             ConfusionMatrixDisplay)

st.title("Breast Cancer Classification - Model Comparison")
st.write("This project compares five machine learning classification models.")

# File names of the five models saved during training.
model_files = {
    "Logistic Regression": "lr.joblib",
    "Decision Tree": "dt.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "nb.joblib",
    "Support Vector Machine": "svm.joblib"
}

# Choose the model from a dropdown.
model_name = st.selectbox("Choose a classification model", list(model_files.keys()))

# Upload a test CSV. The app does not train any model.
uploaded_file = st.file_uploader("Upload test_data.csv", type="csv")

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    feature_names = joblib.load("model/feature_names.joblib")
    required_columns = feature_names + ["target"]
    missing = [col for col in required_columns if col not in test_data.columns]

    if missing:
        st.error("Missing columns: " + ", ".join(missing))
    elif len(test_data) == 0:
        st.error("The CSV is empty.")
    elif test_data[required_columns].isnull().any().any():
        st.error("Please remove or fill missing values in the test CSV.")
    elif not test_data["target"].isin([0, 1]).all():
        st.error("The target must contain 0 (benign) or 1 (malignant).")
    else:
        # Use the same column order that was used for training.
        X_test = test_data[feature_names]
        y_test = test_data["target"]

        model = joblib.load("model/" + model_files[model_name])
        predictions = model.predict(X_test)
        # Both are valid inputs for calculating ROC AUC.
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
        else:
            scores = None

        # Calculate the six required evaluation metrics.
        accuracy = accuracy_score(y_test, predictions)
        auc = roc_auc_score(y_test, scores) if y_test.nunique() == 2 and scores is not None else None
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        mcc = matthews_corrcoef(y_test, predictions)

        st.subheader("Evaluation results")
        metrics = {"Accuracy": accuracy, "AUC": auc, "Precision": precision,
                   "Recall": recall, "F1": f1, "MCC": mcc}
        st.dataframe(pd.DataFrame({"Metric": list(metrics.keys()),
                                   "Score": [round(v, 4) if v is not None else "N/A"
                                             for v in metrics.values()]}), hide_index=True)

        st.subheader("Confusion matrix")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, predictions,
            display_labels=["Benign", "Malignant"], ax=ax, colorbar=False)
        ax.set_title(model_name)
        st.pyplot(fig)
        st.write("Target: 0 = Benign, 1 = Malignant")

        # Show the comparison calculated from the original test split.
        st.subheader("Original test-set comparison (all five models)")
        st.dataframe(pd.read_csv("model_comparison.csv"), hide_index=True)
else:
    st.write("Please upload the provided test_data.csv to see the results.")
