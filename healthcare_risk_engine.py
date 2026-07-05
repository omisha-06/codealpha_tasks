import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Synthesize a Realistic Clinical Patient Registry
np.random.seed(42)
n_patients = 500

# Generating underlying clinical factors
age = np.random.normal(55, 12, n_patients).astype(int)
avg_glucose = np.random.normal(105, 35, n_patients)
bmi = np.random.normal(28, 6, n_patients)
hypertension = np.random.choice([0, 1], size=n_patients, p=[0.75, 0.25])
heart_disease = np.random.choice([0, 1], size=n_patients, p=[0.85, 0.15])
smoking_status = np.random.choice([0, 1, 2], size=n_patients, p=[0.4, 0.3, 0.3]) # 0: Never, 1: Formerly, 2: Smokes

# Logistically calculating custom risk probability based on clinical vectors
risk_score = (age * 0.04) + (avg_glucose * 0.01) + (bmi * 0.03) + (hypertension * 1.2) + (heart_disease * 1.5) + (smoking_status * 0.5)
probability = 1 / (1 + np.exp(-(risk_score - 4.5))) # Sigmoid mapping
stroke_risk = np.random.binomial(1, probability)

# Constructing the Pandas Framework
df = pd.DataFrame({
    'PatientID': range(5001, 5001 + n_patients),
    'Age': np.clip(age, 18, 90),
    'AvgGlucoseLevel': np.clip(avg_glucose, 60, 280),
    'BMI': np.clip(bmi, 15, 50),
    'Hypertension': hypertension,
    'HeartDisease': heart_disease,
    'SmokingStatus': smoking_status,
    'StrokeRiskEvent': stroke_risk
})

# 2. Preprocessing & Splitting Vectors
features = ['Age', 'AvgGlucoseLevel', 'BMI', 'Hypertension', 'HeartDisease', 'SmokingStatus']
X = df[features]
y = df['StrokeRiskEvent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize numerical continuum data scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Engineering: Random Forest Architecture
model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# 4. Evaluation Suite Execution
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"--- Healthcare Risk Engine Diagnostic Accuracy: {accuracy * 100:.2f}% ---")
print("\n--- Clinical Classification Matrix Framework Report ---")
print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))

# 5. Dual-Panel Analytics Plot Generation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot A: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Low Risk', 'High Risk'], yticklabels=['Low Risk', 'High Risk'], ax=ax1)
ax1.set_title('Diagnostic Confusion Matrix', fontsize=12, fontweight='bold')
ax1.set_ylabel('Actual Status')
ax1.set_xlabel('Predicted Risk Status')

# Plot B: Feature Importance Weights
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
sorted_features = [features[i] for i in indices]

sns.barplot(x=importances[indices], y=sorted_features, palette='rocket', ax=ax2)
ax2.set_title('Clinical Risk Factor Importance Weights', fontsize=12, fontweight='bold')
ax2.set_xlabel('Relative Influence Scalar')

plt.tight_layout()
plt.savefig('healthcare_risk_diagnostics.png', dpi=300)
print("\n[Success] Risk analysis complete. Diagnostics layout saved as 'healthcare_risk_diagnostics.png'.")
plt.show()