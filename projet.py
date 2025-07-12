import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Grade Point Average(GPA) Predictor", layout="wide")
st.title("Student GPA Predictor – All Steps Included")

# STEP 1: Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r'C:\Users\ASUS\Desktop\Gomycode\MLformation\StudentsPerformance_with_headers.csv')
    df.drop(columns=["STUDENT ID", "COURSE ID"], inplace=True)
    df.rename(columns={
        'Cumulative grade point average in the last semester (/4.00)': 'GPA_Last',
        'Expected Cumulative grade point average in the graduation (/4.00)': 'GPA_Expected',
        'Do you have a partner': 'HasPartner',
        'Total salary if available': 'Salary',
        'Graduated high-school type': 'HighSchool',
        'Regular artistic or sports activity': 'SportsActivity',
        'Preparation to midterm exams 1': 'PrepMidterm1',
        'Preparation to midterm exams 2': 'PrepMidterm2'
    }, inplace=True)
    return df

df = load_data()
st.success("Data loaded successfully!")

# Show data preview
if st.checkbox(" Show raw data"):
    st.dataframe(df.head())

# STEP 2: Visualizations
st.subheader("Data Visualization")

# Distribution
fig1, ax1 = plt.subplots()
sns.histplot(df['GPA_Last'], kde=True, bins=10, color='skyblue', ax=ax1)
ax1.set_title("GPA_Last Distribution")
st.pyplot(fig1)

# Correlation Heatmap
fig2, ax2 = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
ax2.set_title("Correlation Heatmap")
st.pyplot(fig2)

# STEP 3: Preprocessing
st.subheader(" Train the Model")

# Features and target
target = "GPA_Last"
X = df.drop(columns=["GPA_Last", "GPA_Expected", "GRADE"])
y = df["GPA_Last"]

categorical_cols = ['Sex', 'HighSchool', 'Scholarship type', 'Additional work',
                    'SportsActivity', 'HasPartner', 'Transportation to the university',
                    'Accommodation type in Cyprus', 'Parental status', 'Mother’s occupation',
                    'Father’s occupation', 'PrepMidterm1', 'PrepMidterm2', 'Flip-classroom']

# Preprocessing 
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
pipeline.fit(X_train, y_train)

# Predict 
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

st.markdown(f"""
**Model Performance**
- MAE: `{mae:.2f}`
- RMSE: `{rmse:.2f}`
- R² Score: `{r2:.2f}`
""")

# STEP 4: Prediction form
st.subheader("Predict Grade Point Average for a New Student")

with st.form("predict_form"):
    age = st.selectbox("Student Age", [1, 2, 3])
    sex = st.selectbox("Sex (1=Male, 2=Female)", [1, 2])
    highschool = st.selectbox("High School Type", [1, 2, 3])
    scholarship = st.selectbox("Scholarship Type", [1, 2, 3, 4, 5])
    work = st.selectbox("Additional Work", [1, 2])
    sports = st.selectbox("Regular Sports Activity", [1, 2])
    partner = st.selectbox("Has Partner", [1, 2])
    salary = st.slider("Salary", 1, 5, 1)
    transport = st.selectbox("Transportation", [1, 2, 3, 4])
    accommodation = st.selectbox("Accommodation", [1, 2, 3, 4])
    parent_status = st.selectbox("Parental Status", [1, 2])
    mother_job = st.selectbox("Mother’s Job", [1, 2, 3, 4])
    father_job = st.selectbox("Father’s Job", [1, 2, 3, 4])
    prep1 = st.selectbox("Preparation for Midterm 1", [1, 2, 3])
    prep2 = st.selectbox("Preparation for Midterm 2", [1, 2, 3])
    flip = st.selectbox("Flip-classroom", [1, 2, 3])

    submit = st.form_submit_button(" Predict Grade Point Average")

if submit:
    input_dict = {
    'Student Age': [age],
    'Sex': [sex],
    'HighSchool': [highschool],
    'Scholarship type': [scholarship],
    'Additional work': [work],
    'SportsActivity': [sports],
    'HasPartner': [partner],
    'Salary': [salary],
    'Transportation to the university': [transport],
    'Accommodation type in Cyprus': [accommodation],
    'Mother’s education': [1],  
    'Father’s education ': [1], 
    'Number of sisters/brothers': [1],
    'Parental status': [parent_status],
    'Mother’s occupation': [mother_job],
    'Father’s occupation': [father_job],
    'Weekly study hours': [1],
    'Reading frequency': [1],
    'Reading frequency.1': [1],
    'Attendance to the seminars/conferences related to the department': [1],
    'Impact of your projects/activities on your success': [1],
    'Discussion improves my interest and success in the course': [1],
    'Attendance to classes': [1],
    'PrepMidterm1': [prep1],
    'PrepMidterm2': [prep2],
    'Taking notes in classes': [1],
    'Listening in classes': [1],
    'Flip-classroom': [flip]
}

input_df = pd.DataFrame(input_dict)
prediction = pipeline.predict(input_df)

st.success(f"Predicted Grade Point Average (Last Semester): **{prediction[0]:.2f}** / 5.00")


