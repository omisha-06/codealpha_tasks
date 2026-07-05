import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Synthesize a Realistic Real Estate Market Dataset
np.random.seed(101)
n_houses = 600

# Generating underlying spatial and structural pricing vectors
sqft = np.random.normal(2100, 600, n_houses).astype(int)
bedrooms = np.clip(np.round(sqft / 600 + np.random.normal(1, 0.5, n_houses)), 1, 5).astype(int)
safety_index = np.random.uniform(50, 100, n_houses) # Score out of 100
school_rating = np.random.choice([2, 3, 4, 5], size=n_houses, p=[0.1, 0.3, 0.4, 0.2]) # Star rating
distance_to_cbd = np.random.uniform(2, 25, n_houses) # Distance to central business district in km

# Target Variable Calculation (Base value + feature weights + market noise)
base_price = 80000
market_price = (base_price + 
                (sqft * 130) + 
                (bedrooms * 15000) + 
                (safety_index * 800) + 
                (school_rating * 12000) - 
                (distance_to_cbd * 3200) + 
                np.random.normal(0, 20000, n_houses))

# Constructing the Pandas Dataframe
df = pd.DataFrame({
    'PropertyID': range(9001, 9001 + n_houses),
    'SquareFootage': np.clip(sqft, 600, 5000),
    'Bedrooms': bedrooms,
    'NeighborhoodSafetyIndex': safety_index,
    'SchoolDistrictRating': school_rating,
    'DistanceToCBD_KM': distance_to_cbd,
    'MarketValue_USD': np.clip(market_price, 50000, 950000)
})

# 2. Preprocessing & Validation Framework Split
features = ['SquareFootage', 'Bedrooms', 'NeighborhoodSafetyIndex', 'SchoolDistrictRating', 'DistanceToCBD_KM']
X = df[features]
y = df['MarketValue_USD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical ranges for model optimization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Engineering: Gradient Boosting Regressor
model = GradientBoostingRegressor(n_estimators=120, max_depth=4, learning_rate=0.08, random_state=42)
model.fit(X_train_scaled, y_train)

# 4. Valuation Metrics Calculation
predictions = model.predict(X_test_scaled)
r2 = r2_score(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"--- Real Estate Predictive Valuation Accuracy (R² Score): {r2 * 100:.2f}% ---")
print(f"--- Root Mean Squared Error (RMSE): ${rmse:,.2f} ---")

# 5. Dashboard Visualization Output Generation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot A: Actual Values vs Model Predictions
sns.scatterplot(x=y_test, y=predictions, color='#4D96FF', alpha=0.6, ax=ax1)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='#FF6B6B')
ax1.set_title('Actual Market Values vs. Predicted Appraisals', fontsize=12, fontweight='bold')
ax1.set_xlabel('Actual Valuation ($)')
ax1.set_ylabel('Model Predicted Valuation ($)')

# Plot B: Feature Importance Weights
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
sorted_features = [features[i] for i in indices]

sns.barplot(x=importances[indices], y=sorted_features, palette='viridis', ax=ax2)
ax2.set_title('Structural Property Value Driver Influences', fontsize=12, fontweight='bold')
ax2.set_xlabel('Relative Valuation Predictive Weight')

plt.tight_layout()
plt.savefig('real_estate_valuation_analytics.png', dpi=300)
print("\n[Success] Valuation complete. Market insights saved as 'real_estate_valuation_analytics.png'.")
plt.show()