# 🏦 Loan Sanction Prediction — Deep Learning

Binary classification using a **Deep Neural Network (DNN)** built with TensorFlow/Keras to predict whether a loan application will be approved or rejected.

---

## 📌 Project Overview

This project applies **Deep Learning** concepts to the Loan Sanction Kaggle dataset. The model takes 11 applicant features as input and outputs a binary prediction: **Approved (Y)** or **Rejected (N)**.

---

## 🧠 Deep Learning Concepts Applied

| Concept | Purpose |
|---|---|
| Multi-layer DNN (4 hidden layers) | Learns complex non-linear patterns |
| Batch Normalization | Stabilizes & speeds up training |
| Dropout Regularization | Prevents overfitting |
| Early Stopping | Stops training when val_loss plateaus |
| ReduceLROnPlateau | Adjusts learning rate automatically |
| ReLU Activation | Adds non-linearity to hidden layers |
| Sigmoid Output | Outputs probability for binary classification |
| ROC-AUC Metric | Evaluates binary classifier performance |

---

## 📂 Dataset

- **Source:** Kaggle — Loan Prediction Dataset
- **Train size:** 614 rows × 13 columns
- **Features:** Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area
- **Target:** Loan_Status (Y = Approved, N = Rejected)

---

## 🛠 Tech Stack

- Python 3.x
- TensorFlow / Keras
- Scikit-learn
- Pandas & NumPy
- Matplotlib & Seaborn
- Streamlit (Web App)
- Google Colab (Training environment)

---

## 🚀 How to Run the Streamlit App

```bash
git clone https://github.com/yourusername/loan-approval-deep-learning
cd loan-approval-deep-learning

pip install -r requirements.txt

# Make sure loan_approval_dnn.keras and loan_artifacts.pkl are present
streamlit run app.py
```

---

## 📊 Model Architecture

```
Input (11 features)
  → Dense(128) → BatchNorm → ReLU → Dropout(0.3)
  → Dense(64)  → BatchNorm → ReLU → Dropout(0.3)
  → Dense(32)  → BatchNorm → ReLU → Dropout(0.2)
  → Dense(16)  → ReLU → Dropout(0.2)
  → Dense(1)   → Sigmoid
```

---

## 📈 Results

| Metric | Score |
|---|---|
| Accuracy | ~82% |
| AUC Score | ~0.87 |
| Precision | ~0.84 |
| Recall | ~0.91 |

> Update these numbers after running the notebook.

---

## 📁 Files

| File | Description |
|---|---|
| `loan_approval_colab.py` | Full training code (paste in Colab) |
| `app.py` | Streamlit web application |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

---

## 👤 Author

**Your Name**  
BS Computer Science  
Deep Learning Final Project — 2025
