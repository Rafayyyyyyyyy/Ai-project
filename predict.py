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
try:
    car_data = pd.read_csv(file_path)
except FileNotFoundError:
    st.error("The dataset file 'honda_car_selling.csv' was not found. Please upload the file and try again.")
    st.stop()

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

# User inputs for car model and fuel type
model_options = car_data['Car Model'].unique()
selected_model = st.selectbox("Select the car model:", model_options)

fuel_options = car_data['Fuel Type'].unique()
selected_fuel = st.radio("Select the fuel type:", fuel_options)

# Filter the data based on user selection
filtered_data = car_data[(car_data['Car Model'] == selected_model) & (car_data['Fuel Type'] == selected_fuel)]

if filtered_data.empty:
    st.error("No data available for the selected model and fuel type.")
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

# Bar graph of average price by year
st.subheader("Average Price by Year")
avg_price_by_year = car_data.groupby('Year')['Price'].mean().sort_index()
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_price_by_year.index, y=avg_price_by_year.values)
plt.title("Average Price by Year")
plt.xlabel("Year")
plt.ylabel("Average Price (Lakh)")
st.pyplot(plt)

# Generalized bar chart of average price by car model
st.subheader("Generalized Average Price by Car Model")
general_avg_price_by_model = car_data.groupby('Car Model')['Price'].mean().sort_values()
plt.figure(figsize=(6, 4))
sns.barplot(x=general_avg_price_by_model.values, y=general_avg_price_by_model.index, palette='muted')
plt.title("Generalized Average Price by Car Model")
plt.xlabel("Average Price (Lakh)")
plt.ylabel("Car Model")
st.pyplot(plt)

# Pie chart of car distribution by fuel type
st.subheader("Car Distribution by Fuel Type")
fuel_type_distribution = car_data['Fuel Type'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(fuel_type_distribution, labels=fuel_type_distribution.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title("Car Distribution by Fuel Type")
st.pyplot(plt)

# Smaller generalized pie chart of car distribution by model
st.subheader("Generalized Car Distribution by Model")
car_model_distribution = car_data['Car Model'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(car_model_distribution, labels=car_model_distribution.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title("Car Distribution by Model")
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
