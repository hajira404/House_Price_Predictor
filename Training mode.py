import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Step 2: Load the Dataset
data = pd.read_csv('train.csv')

# Step 3.1: Handling Missing Values
bathroom_features = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
data[bathroom_features] = data[bathroom_features].fillna(0)
data['BedroomAbvGr'] = data['BedroomAbvGr'].fillna(data['BedroomAbvGr'].mode()[0])

# Step 3.2: Feature Engineering
data['TotalBathrooms'] = (
    data['FullBath'] +
    data['HalfBath'] * 0.5 +
    data['BsmtFullBath'] +
    data['BsmtHalfBath'] * 0.5
)

# Step 3.3: Selecting Features and Target
features = ['GrLivArea', 'BedroomAbvGr', 'TotalBathrooms']
X = data[features]
y = data['SalePrice']

# Step 4: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=features)

# Step 5: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Evaluate the Model
y_pred = model.predict(X_test)
def mean_squared_error_manual(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

mse = mean_squared_error_manual(y_test, y_pred)
print(f'\nMean Squared Error (MSE): {mse:.2f}')

rmse = np.sqrt(mse)
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

r2 = r2_score(y_test, y_pred)
print(f'R-squared (R²): {r2:.2f}')

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Perfect prediction line
plt.show()

# Save the model and scaler to a .pkl file
with open('model_and_scaler.pkl', 'wb') as file:
    pickle.dump((model, scaler,r2), file)

print(f'Model and scaler saved to model_and_scaler.pkl')