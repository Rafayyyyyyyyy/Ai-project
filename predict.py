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

# Line chart for price against year
st.subheader("Price by Year")
# Group data by 'Year' and calculate the mean price for each year
avg_price_by_year = car_data.groupby('Year')['Price'].mean().sort_index()

# Plotting the line chart for price by year
plt.figure(figsize=(10, 6))
sns.lineplot(x=avg_price_by_year.index, y=avg_price_by_year.values, marker='o', color='b')
plt.title("Price by Year")
plt.xlabel("Year")
plt.ylabel("Average Price (Lakh)")
plt.grid(True)
st.pyplot(plt)

# Generalized pie chart of kilometers driven ranges
st.subheader("Distribution of Kilometers Driven")
# Create bins for kms driven
kms_bins = [0, 20000, 40000, 60000, 80000, 100000, 120000, 150000, 200000]
kms_labels = ['0-20k', '20k-40k', '40k-60k', '60k-80k', '80k-100k', '100k-120k', '120k-150k', '150k-200k']
car_data['kms_range'] = pd.cut(car_data['kms Driven'], bins=kms_bins, labels=kms_labels, right=False)

# Pie chart for kms driven ranges
kms_distribution = car_data['kms_range'].value_counts()
plt.figure(figsize=(8, 8))
kms_distribution.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set3', len(kms_distribution)))
plt.title("Kilometers Driven Distribution")
st.pyplot(plt)

# Bar chart for distribution by Year
st.subheader("Car Distribution by Year")
# Count the number of cars for each year
year_distribution = car_data['Year'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
sns.barplot(x=year_distribution.index, y=year_distribution.values, palette='viridis')
plt.title("Car Distribution by Year")
plt.xlabel("Year")
plt.ylabel("Count of Cars")
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

