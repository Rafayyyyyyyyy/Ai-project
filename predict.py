import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import streamlit as st

# Streamlit app
st.title("Car Price Prediction App")

# Load the dataset
file_path = 'honda_car_selling.csv'
car_data = pd.read_csv(file_path)

# Data cleaning
car_data['kms Driven'] = car_data['kms Driven'].str.replace(' kms', '').str.replace(',', '').astype(float)
car_data['Price'] = car_data['Price'].str.replace(' Lakh', '').str.replace(',', '').astype(float)

# Remove outliers
q1_kms = car_data['kms Driven'].quantile(0.25)
q3_kms = car_data['kms Driven'].quantile(0.75)
iqr_kms = q3_kms - q1_kms
lower_bound_kms = q1_kms - 1.5 * iqr_kms
upper_bound_kms = q3_kms + 1.5 * iqr_kms

q1_price = car_data['Price'].quantile(0.25)
q3_price = car_data['Price'].quantile(0.75)
iqr_price = q3_price - q1_price
lower_bound_price = q1_price - 1.5 * iqr_price
upper_bound_price = q3_price + 1.5 * iqr_price

car_data = car_data[(car_data['kms Driven'] >= lower_bound_kms) & (car_data['kms Driven'] <= upper_bound_kms)]
car_data = car_data[(car_data['Price'] >= lower_bound_price) & (car_data['Price'] <= upper_bound_price)]

# User inputs for car model and transmission type
model_options = car_data['Car Model'].unique()
selected_model = st.selectbox("Select the car model:", model_options)

transmission_options = car_data['Suspension'].unique()
selected_transmission = st.radio("Select the transmission type:", transmission_options)

# Filter the data based on user selection
filtered_data = car_data[(car_data['Car Model'] == selected_model) & (car_data['Suspension'] == selected_transmission)]

if filtered_data.empty:
    st.error("No data available for the selected model and transmission type.")
    st.stop()

# Define features and target variable
X = filtered_data[['kms Driven']]
y = filtered_data['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions for evaluation
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
st.write(f"Mean Squared Error: {mse}")

# Visualize the results using matplotlib
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.scatter(X_test, y_pred, color='red', label='Predicted Prices')
plt.plot(X_test, y_pred, color='red', linestyle='dashed', label='Regression Line')
plt.title('Car Price Prediction')
plt.xlabel('Kilometers Driven')
plt.ylabel('Price (Lakh)')
plt.legend()
st.pyplot(plt)

# Bar graph of average price by model
st.subheader("Average Price by Car Model")
avg_price_by_model = car_data.groupby('Car Model')['Price'].mean().sort_values()
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_price_by_model.index, y=avg_price_by_model.values)
plt.xticks(rotation=45)
plt.title("Average Price by Car Model")
plt.xlabel("Car Model")
plt.ylabel("Average Price (Lakh)")
st.pyplot(plt)

# User input for prediction
user_input = st.number_input("Enter the kilometers driven:", min_value=0.0, step=100.0)

if user_input > 0:
    # Calculate predicted price based on kilometers driven
    predicted_price = model.predict(pd.DataFrame([[user_input]], columns=['kms Driven']))[0]
    # Ensure price does not get negative
    adjusted_price = max(predicted_price, 0.0)
    st.write(f"Predicted Price: {adjusted_price:.2f} Lakh")

# Display raw data option
if st.checkbox("Show raw data"):
    st.write(car_data)

