import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Example of fixing the import redundancy
from joblib import dump, load  # Import joblib only once and use specific functions if needed
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv('house_prices.csv')
X = data[['feature1', 'feature2']]
y = data['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save model

joblib.dump(model, 'model.joblib')
