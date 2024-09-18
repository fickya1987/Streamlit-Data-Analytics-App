import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Step 1: Load and View Data
st.title("Google Play Store Apps Data Analysis")

# Uploading the file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("### Data Preview:")
    st.write(data.head())

    # Step 2: Data Cleaning
    st.write("### Data Cleaning")
    
    # Drop duplicates
    data = data.drop_duplicates()
    
    # Handle missing values - here we simply drop them
    data = data.dropna()

    st.write("Data after cleaning:")
    st.write(data.head())

    # Step 3: Data Visualization
    st.write("### Data Visualization")
    
    fig, ax = plt.subplots()
    sns.countplot(x='Category', data=data, ax=ax)
    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    sns.scatterplot(x='Rating', y='Reviews', data=data, ax=ax)
    st.pyplot(fig)

    # Step 4: Machine Learning - Linear Regression and Decision Tree

    # Selecting features and target
    data['Reviews'] = data['Reviews'].astype(int)  # Ensure 'Reviews' is integer type
    features = data[['Reviews']]
    target = data['Rating']
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    
    # Linear Regression Model
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lr = lin_reg.predict(X_test)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    
    # Decision Tree Model
    dt_reg = DecisionTreeRegressor(random_state=42)
    dt_reg.fit(X_train, y_train)
    y_pred_dt = dt_reg.predict(X_test)
    mse_dt = mean_squared_error(y_test, y_pred_dt)
    
    # Displaying the results
    st.write("### Model Results")
    
    st.write(f"Linear Regression MSE: {mse_lr:.2f}")
    fig, ax = plt.subplots()
    plt.scatter(y_test, y_pred_lr)
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title('Linear Regression: Actual vs Predicted')
    st.pyplot(fig)
    
    st.write(f"Decision Tree MSE: {mse_dt:.2f}")
    fig, ax = plt.subplots()
    plt.scatter(y_test, y_pred_dt)
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title('Decision Tree: Actual vs Predicted')
    st.pyplot(fig)

