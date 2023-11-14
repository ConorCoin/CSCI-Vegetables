import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score

pd.set_option('display.max_rows', None)
# Assuming you have a CSV file named 'concrete_data.csv' with the dataset
# Replace this with the actual path and file name if different
data = pd.read_csv('Vegetable_market.csv')
print(data)


# Assuming features are in X and target (Price per Kilogram) is in y
X = data[['Season', 'Month', 'Temp', 'Deasaster Happen in last 3month', 'Vegetable condition']]
y = data['Price per kg']

# Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = linear_reg_model.predict(X_test)

# Calculate regression metrics
r2_value = r2_score(y_test, y_pred)
mean_squared_error_value = mean_squared_error(y_test, y_pred)

# Visualization of Predictions vs Actual Values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Linear Regression: Actual vs Predicted Prices")
plt.show()

# Print the results
print("R2 Value:", r2_value)
print("Mean Squared Error:", mean_squared_error_value)
