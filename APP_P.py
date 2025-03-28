import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Set plot style
sns.set()

#########################
# Helper Functions
#########################

def load_and_split_regression():
    # Regression dataset: Diabetes
    data = load_diabetes()
    X, y = data.data, data.target
    return train_test_split(X, y, test_size=0.2, random_state=42)

def load_and_split_classification():
    # Classification dataset: Iris
    iris = load_iris()
    X, y = iris.data, iris.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=False,
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    st.pyplot(plt.gcf())
    plt.clf()

def plot_regression_results(y_test, y_pred):
    plt.figure(figsize=(6,4))
    plt.scatter(y_test, y_pred, color='navy', alpha=0.7)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("True vs Predicted")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
    st.pyplot(plt.gcf())
    plt.clf()

#########################
# ML Model Functions
#########################

# Regression: Linear Regression
def run_linear_regression():
    X_train, X_test, y_train, y_test = load_and_split_regression()
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse, y_test, y_pred

# Classification: Logistic Regression with options
def run_logistic_regression(penalty='l2', solver='lbfgs', l1_ratio=None):
    X_train, X_test, y_train, y_test = load_and_split_classification()
    if penalty == 'elasticnet':
        model = LogisticRegression(penalty=penalty, solver='saga', l1_ratio=l1_ratio, max_iter=1000, multi_class='auto')
    else:
        model = LogisticRegression(penalty=penalty, solver=solver, max_iter=1000, multi_class='auto')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return acc, cm, report

def run_knn():
    X_train, X_test, y_train, y_test = load_and_split_classification()
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return acc, cm, report

def run_naive_bayes():
    X_train, X_test, y_train, y_test = load_and_split_classification()
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return acc, cm, report

def run_svm():
    X_train, X_test, y_train, y_test = load_and_split_classification()
    model = SVC(kernel='rbf', gamma='scale')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return acc, cm, report

def run_decision_tree():
    X_train, X_test, y_train, y_test = load_and_split_classification()
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return acc, cm, report

def run_random_forest():
    X_train, X_test, y_train, y_test = load_and_split_classification()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return acc, cm, report

#########################
# Streamlit Interactive UI
#########################

st.title("Interactive ML Model Trainer & Evaluator")
st.write("This app trains several ML models using Scikit-Learn and visualizes the evaluation metrics.")
st.title(" Kumar Harsh, Roll No. : 2205137 ")

# Sidebar to choose model type: Regression or Classification
model_type = st.sidebar.radio("Select Model Type", ("Regression", "Classification"))

if model_type == "Regression":
    st.sidebar.subheader("Regression: Linear Regression")
    if st.sidebar.button("Run Linear Regression"):
        st.subheader("Linear Regression on Diabetes Dataset")
        mse, y_test, y_pred = run_linear_regression()
        st.write(f"**Mean Squared Error:** {mse:.2f}")
        st.write("### Regression Plot (True vs Predicted)")
        plot_regression_results(y_test, y_pred)
        
elif model_type == "Classification":
    st.sidebar.subheader("Classification Models on Iris Dataset")
    classifier = st.sidebar.selectbox("Select Classifier", 
                                      ["Logistic Regression (L2)", 
                                       "Logistic Regression (L1)",
                                       "Logistic Regression (Elastic Net)",
                                       "K-Nearest Neighbors",
                                       "Naive Bayes",
                                       "Support Vector Machine",
                                       "Decision Tree",
                                       "Random Forest"])
    if st.sidebar.button("Run Classifier"):
        st.subheader(f"Results for {classifier}")
        if classifier == "Logistic Regression (L2)":
            acc, cm, report = run_logistic_regression(penalty='l2', solver='lbfgs')
        elif classifier == "Logistic Regression (L1)":
            acc, cm, report = run_logistic_regression(penalty='l1', solver='liblinear')
        elif classifier == "Logistic Regression (Elastic Net)":
            acc, cm, report = run_logistic_regression(penalty='elasticnet', solver='saga', l1_ratio=0.5)
        elif classifier == "K-Nearest Neighbors":
            acc, cm, report = run_knn()
        elif classifier == "Naive Bayes":
            acc, cm, report = run_naive_bayes()
        elif classifier == "Support Vector Machine":
            acc, cm, report = run_svm()
        elif classifier == "Decision Tree":
            acc, cm, report = run_decision_tree()
        elif classifier == "Random Forest":
            acc, cm, report = run_random_forest()
        
        st.write(f"**Accuracy:** {acc*100:.2f}%")
        st.write("### Confusion Matrix")
        class_names = load_iris().target_names
        plot_confusion_matrix(cm, classes=class_names)
        st.write("### Classification Report")
        report_df = pd.DataFrame(report).transpose().round(2)
        st.dataframe(report_df)
