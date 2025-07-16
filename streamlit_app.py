import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier



df = pd.read_csv("data.csv")
st.title("ML Classifier Comparison App")
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)

x = df.drop(columns=["diagnosis"])  # assumes last column is target
y = df["diagnosis"]
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
le = LabelEncoder()
y = le.fit_transform(y)

models = {
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': GaussianNB(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

classifier_name = st.selectbox("Choose a Classifier", list(models.keys()))
model = models[classifier_name]


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('Scaler', MinMaxScaler()),
    ('Classifier', model)
])


pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

scores = cross_validate(
    pipeline, x, y, cv=5,
    scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
    return_train_score=True
)

st.subheader("Evaluation Metrics")
st.write(f"**Train Accuracy**: {scores['train_accuracy'].mean():.4f}")
st.write(f"**Test Accuracy**: {scores['test_accuracy'].mean():.4f}")
st.write(f"**Precision**: {scores['test_precision_weighted'].mean():.4f}")
st.write(f"**Recall**: {scores['test_recall_weighted'].mean():.4f}")
st.write(f"**F1 Score**: {scores['test_f1_weighted'].mean():.4f}")


st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test, cmap='Blues', ax=ax)
st.pyplot(fig)


st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))
