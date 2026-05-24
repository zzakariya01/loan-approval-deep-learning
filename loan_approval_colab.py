# ============================================================
# DEEP LEARNING FINAL PROJECT
# Loan Sanction / Approval Prediction
# Dataset: loan_sanction_train.csv / loan_sanction_test.csv
# Model: Deep Neural Network (TensorFlow / Keras)
# ============================================================


# ──────────────────────────────────────────────
# CELL 1 ▸ Install all required libraries
# ──────────────────────────────────────────────
# Paste this as the VERY FIRST cell and run it once

!pip install tensorflow scikit-learn pandas numpy \
            matplotlib seaborn streamlit pyngrok \
            imbalanced-learn -q

print("✅ All libraries installed successfully!")


# ──────────────────────────────────────────────
# CELL 2 ▸ Import libraries
# ──────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, accuracy_score
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)

print("TensorFlow version :", tf.__version__)
print("NumPy version      :", np.__version__)
print("Pandas version     :", pd.__version__)


# ──────────────────────────────────────────────
# CELL 3 ▸ Upload and load datasets
# ──────────────────────────────────────────────
# Run this cell → click "Choose Files" → select BOTH CSVs

from google.colab import files
print("Upload your CSV files now...")
uploaded = files.upload()   # select loan_sanction_train.csv AND loan_sanction_test.csv

train_df = pd.read_csv('loan_sanction_train.csv')
test_df  = pd.read_csv('loan_sanction_test.csv')

print(f"\n✅ Train shape : {train_df.shape}")
print(f"✅ Test shape  : {test_df.shape}")
print("\n--- First 5 rows of training data ---")
print(train_df.head())


# ──────────────────────────────────────────────
# CELL 4 ▸ Exploratory Data Analysis (EDA)
# ──────────────────────────────────────────────

print("=== Dataset Info ===")
print(train_df.info())
print("\n=== Missing Values ===")
print(train_df.isnull().sum())
print("\n=== Target Distribution ===")
print(train_df['Loan_Status'].value_counts())

# Plot 1 – Target distribution
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Exploratory Data Analysis — Loan Sanction Dataset',
             fontsize=14, fontweight='bold')

colors = ['#E24B4A', '#1D9E75']
train_df['Loan_Status'].value_counts().plot(
    kind='bar', ax=axes[0, 0], color=colors, edgecolor='white')
axes[0, 0].set_title('Loan Approval Distribution')
axes[0, 0].set_xlabel('Status  (Y = Approved, N = Rejected)')
axes[0, 0].set_ylabel('Count')
axes[0, 0].tick_params(axis='x', rotation=0)

# Plot 2 – Missing values heatmap
sns.heatmap(train_df.isnull(), cbar=False,
            ax=axes[0, 1], cmap='YlOrRd', yticklabels=False)
axes[0, 1].set_title('Missing Values Heatmap')

# Plot 3 – Applicant Income distribution
sns.histplot(train_df['ApplicantIncome'], bins=40,
             kde=True, ax=axes[0, 2], color='#185FA5')
axes[0, 2].set_title('Applicant Income Distribution')

# Plot 4 – Loan Amount distribution
sns.histplot(train_df['LoanAmount'].dropna(), bins=30,
             kde=True, ax=axes[1, 0], color='#533AB7')
axes[1, 0].set_title('Loan Amount Distribution')

# Plot 5 – Credit History vs Loan Status
ct = pd.crosstab(train_df['Credit_History'],
                 train_df['Loan_Status'])
ct.plot(kind='bar', ax=axes[1, 1],
        color=['#E24B4A', '#1D9E75'], edgecolor='white')
axes[1, 1].set_title('Credit History vs Loan Status')
axes[1, 1].tick_params(axis='x', rotation=0)
axes[1, 1].legend(['Rejected', 'Approved'])

# Plot 6 – Property Area vs Loan Status
ct2 = pd.crosstab(train_df['Property_Area'],
                  train_df['Loan_Status'])
ct2.plot(kind='bar', ax=axes[1, 2],
         color=['#E24B4A', '#1D9E75'], edgecolor='white')
axes[1, 2].set_title('Property Area vs Loan Status')
axes[1, 2].tick_params(axis='x', rotation=0)
axes[1, 2].legend(['Rejected', 'Approved'])

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ EDA plots saved as eda_plots.png")


# ──────────────────────────────────────────────
# CELL 5 ▸ Data Preprocessing
# ──────────────────────────────────────────────

df = train_df.copy()

# 5a – Drop Loan_ID (not a feature)
df.drop('Loan_ID', axis=1, inplace=True)

# 5b – Fix Dependents ('3+' → 3, then convert to int)
df['Dependents'] = df['Dependents'].replace('3+', 3)
df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce')

# 5c – Fill missing values
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

print("Missing after fill:", df.isnull().sum().sum(), "values")

# 5d – Encode categorical columns
le_gender        = LabelEncoder()
le_married       = LabelEncoder()
le_education     = LabelEncoder()
le_self_employed = LabelEncoder()
le_property      = LabelEncoder()
le_target        = LabelEncoder()

df['Gender']        = le_gender.fit_transform(df['Gender'])
df['Married']       = le_married.fit_transform(df['Married'])
df['Education']     = le_education.fit_transform(df['Education'])
df['Self_Employed'] = le_self_employed.fit_transform(df['Self_Employed'])
df['Property_Area'] = le_property.fit_transform(df['Property_Area'])
df['Loan_Status']   = le_target.fit_transform(df['Loan_Status'])  # Y→1, N→0

print("\nLabel encoding done.")
print("Loan_Status classes:", le_target.classes_)   # ['N', 'Y']
print(df.head())

# 5e – Correlation heatmap (after encoding)
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True, fmt='.2f',
            cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# 5f – Feature / target split + scaling
feature_cols = ['Gender', 'Married', 'Dependents', 'Education',
                'Self_Employed', 'ApplicantIncome',
                'CoapplicantIncome', 'LoanAmount',
                'Loan_Amount_Term', 'Credit_History', 'Property_Area']

X = df[feature_cols].values
y = df['Loan_Status'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nX_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")
print(f"Class distribution in train: {np.bincount(y_train)}")


# ──────────────────────────────────────────────
# CELL 6 ▸ Build the Deep Neural Network
# ──────────────────────────────────────────────

def build_dnn(input_dim):
    """
    4-layer Deep Neural Network for binary classification.
    Deep Learning concepts used:
      • Multi-layer architecture (depth)
      • Batch Normalization (training stability)
      • Dropout (regularization / overfitting prevention)
      • ReLU activations (non-linearity)
      • Sigmoid output (binary probability)
      • Adam optimizer (adaptive learning rate)
      • Binary Cross-Entropy loss
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,), name='input'),

        # ── Hidden Layer 1 ──────────────────────
        layers.Dense(128, name='dense_1'),
        layers.BatchNormalization(name='bn_1'),
        layers.Activation('relu'),
        layers.Dropout(0.3, name='dropout_1'),

        # ── Hidden Layer 2 ──────────────────────
        layers.Dense(64, name='dense_2'),
        layers.BatchNormalization(name='bn_2'),
        layers.Activation('relu'),
        layers.Dropout(0.3, name='dropout_2'),

        # ── Hidden Layer 3 ──────────────────────
        layers.Dense(32, name='dense_3'),
        layers.BatchNormalization(name='bn_3'),
        layers.Activation('relu'),
        layers.Dropout(0.2, name='dropout_3'),

        # ── Hidden Layer 4 ──────────────────────
        layers.Dense(16, activation='relu', name='dense_4'),
        layers.Dropout(0.2, name='dropout_4'),

        # ── Output Layer ────────────────────────
        layers.Dense(1, activation='sigmoid', name='output')
    ], name='LoanApproval_DNN')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    return model

model = build_dnn(X_train.shape[1])
model.summary()


# ──────────────────────────────────────────────
# CELL 7 ▸ Callbacks (Early Stopping, LR Scheduler)
# ──────────────────────────────────────────────

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        'best_loan_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print("✅ Callbacks configured:")
print("   • EarlyStopping    – stops if val_loss doesn't improve (patience=20)")
print("   • ReduceLROnPlateau – halves LR if val_loss stalls (patience=8)")
print("   • ModelCheckpoint  – saves best model to best_loan_model.keras")


# ──────────────────────────────────────────────
# CELL 8 ▸ Train the Model
# ──────────────────────────────────────────────

history = model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Training complete!")
print(f"   Epochs run     : {len(history.history['loss'])}")
print(f"   Best val_acc   : {max(history.history['val_accuracy']):.4f}")


# ──────────────────────────────────────────────
# CELL 9 ▸ Plot Training History
# ──────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Model Training History', fontsize=14, fontweight='bold')

metrics = [
    ('accuracy', 'Accuracy'),
    ('loss',     'Loss'),
    ('auc',      'AUC Score'),
]

for ax, (metric, title) in zip(axes, metrics):
    ax.plot(history.history[metric],
            label='Train', color='#185FA5', linewidth=2)
    ax.plot(history.history[f'val_{metric}'],
            label='Validation', color='#E24B4A', linewidth=2)
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
plt.show()


# ──────────────────────────────────────────────
# CELL 10 ▸ Evaluate on Test Set
# ──────────────────────────────────────────────

loss, acc, auc, prec, rec = model.evaluate(X_test, y_test, verbose=0)

y_pred_prob = model.predict(X_test, verbose=0).flatten()
y_pred      = (y_pred_prob >= 0.5).astype(int)

f1 = 2 * (prec * rec) / (prec + rec + 1e-8)

print("=" * 45)
print("        TEST SET EVALUATION RESULTS")
print("=" * 45)
print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  AUC Score : {auc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"  Loss      : {loss:.4f}")
print("=" * 45)

print("\n--- Classification Report ---")
print(classification_report(
    y_test, y_pred,
    target_names=['Rejected (N)', 'Approved (Y)']
))


# ──────────────────────────────────────────────
# CELL 11 ▸ Confusion Matrix
# ──────────────────────────────────────────────

plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Rejected', 'Approved'],
    yticklabels=['Rejected', 'Approved'],
    linewidths=1, linecolor='white'
)
plt.title('Confusion Matrix — Test Set', fontsize=13, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()


# ──────────────────────────────────────────────
# CELL 12 ▸ ROC Curve
# ──────────────────────────────────────────────

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='#185FA5', lw=2.5,
         label=f'DNN  (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1.5,
         linestyle='--', label='Random Classifier')
plt.fill_between(fpr, tpr, alpha=0.08, color='#185FA5')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve — Loan Approval DNN', fontsize=13, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
plt.show()


# ──────────────────────────────────────────────
# CELL 13 ▸ Save Model, Scaler & Encoders
# ──────────────────────────────────────────────

# Save the final Keras model
model.save('loan_approval_dnn.keras')

# Save scaler and encoders (needed by Streamlit app)
artifacts = {
    'scaler':           scaler,
    'le_gender':        le_gender,
    'le_married':       le_married,
    'le_education':     le_education,
    'le_self_employed': le_self_employed,
    'le_property':      le_property,
    'le_target':        le_target,
    'feature_cols':     feature_cols,
}
with open('loan_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("✅ Saved:")
print("   • loan_approval_dnn.keras  ← trained DNN model")
print("   • loan_artifacts.pkl       ← scaler + all encoders")
print("   • best_loan_model.keras    ← best checkpoint (from callback)")


# ──────────────────────────────────────────────
# CELL 14 ▸ Create Streamlit app.py
# ──────────────────────────────────────────────
# Run this cell — it writes app.py to disk automatically

app_code = '''
import streamlit as st
import numpy as np
import pickle
import tensorflow as tf

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("loan_approval_dnn.keras")
    with open("loan_artifacts.pkl", "rb") as f:
        arts = pickle.load(f)
    return model, arts

model, arts = load_artifacts()

# ── Header ────────────────────────────────────────
st.title("🏦 Loan Sanction Predictor")
st.caption("Deep Learning — Binary Classification (DNN)")
st.markdown("---")

# ── Input Form ───────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    gender         = st.selectbox("Gender",          ["Male", "Female"])
    married        = st.selectbox("Married",          ["Yes", "No"])
    dependents     = st.selectbox("Dependents",       ["0", "1", "2", "3+"])
    education      = st.selectbox("Education",        ["Graduate", "Not Graduate"])
    self_employed  = st.selectbox("Self Employed",    ["No", "Yes"])
    credit_history = st.selectbox("Credit History",   [1, 0],
                                   format_func=lambda x: "Good (1)" if x == 1 else "Bad (0)")

with col2:
    applicant_income   = st.number_input("Applicant Income (₹)",
                                          min_value=0, value=5000, step=500)
    coapplicant_income = st.number_input("Co-applicant Income (₹)",
                                          min_value=0, value=0, step=500)
    loan_amount        = st.number_input("Loan Amount (thousands)",
                                          min_value=1, value=150, step=5)
    loan_amount_term   = st.number_input("Loan Term (months)",
                                          min_value=12, value=360, step=12)
    property_area      = st.selectbox("Property Area",
                                       ["Urban", "Semiurban", "Rural"])

st.markdown("---")

# ── Prediction ───────────────────────────────────
if st.button("🔍  Predict Loan Status", use_container_width=True):

    dep_val = 3 if dependents == "3+" else int(dependents)

    # Encode using the same LabelEncoders from training
    gender_enc   = arts["le_gender"].transform([gender])[0]
    married_enc  = arts["le_married"].transform([married])[0]
    edu_enc      = arts["le_education"].transform([education])[0]
    se_enc       = arts["le_self_employed"].transform([self_employed])[0]
    prop_enc     = arts["le_property"].transform([property_area])[0]

    features = np.array([[gender_enc, married_enc, dep_val,
                           edu_enc, se_enc,
                           applicant_income, coapplicant_income,
                           loan_amount, loan_amount_term,
                           credit_history, prop_enc]], dtype=float)

    features_scaled = arts["scaler"].transform(features)

    prob      = float(model.predict(features_scaled, verbose=0)[0][0])
    approved  = prob >= 0.5

    st.subheader("📊 Prediction Result")

    if approved:
        st.success(f"✅  APPROVED — Confidence: {prob*100:.1f}%")
    else:
        st.error(f"❌  REJECTED — Confidence: {(1-prob)*100:.1f}%")

    st.progress(float(prob))
    st.caption(f"Approval probability score: {prob:.4f}")

    with st.expander("ℹ️ About this model"):
        st.markdown("""
        **Model:** Deep Neural Network (4 hidden layers)  
        **Concepts:** Batch Normalization · Dropout · Early Stopping  
        · Learning Rate Scheduler · ROC-AUC evaluation  
        **Dataset:** Kaggle — Loan Sanction Dataset  
        **Framework:** TensorFlow / Keras
        """)
'''

with open('app.py', 'w') as f:
    f.write(app_code)

print("✅ app.py created successfully!")


# ──────────────────────────────────────────────
# CELL 15 ▸ Launch Streamlit app inside Colab
# ──────────────────────────────────────────────
# Run this LAST — it opens a public URL for your app

import subprocess, time
from pyngrok import ngrok

# Kill any existing ngrok tunnels
ngrok.kill()

# Start Streamlit server in background
proc = subprocess.Popen(
    ['streamlit', 'run', 'app.py',
     '--server.port', '8501',
     '--server.headless', 'true',
     '--browser.gatherUsageStats', 'false'],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

# Wait for server to start
time.sleep(4)

# Open public ngrok tunnel
public_url = ngrok.connect(8501)
print("\n" + "="*50)
print("  ✅ YOUR STREAMLIT APP IS LIVE!")
print(f"  🔗 Open this URL: {public_url}")
print("="*50)
print("\n  Share this URL for your demo / screenshots.")
print("  Press [ ■ Stop ] in Colab to shut down the server.\n")
