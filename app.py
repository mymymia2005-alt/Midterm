import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


st.set_page_config(
    page_title="Student Performance Prediction App",
    layout="wide"
)

st.title("Student Performance Prediction App")


df = pd.read_csv("student_data.csv", sep=";")

df = df.dropna()

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a page:",
    [
        "Business Case & Data Presentation",
        "Data Visualization",
        "Prediction Model"
    ]
)

# Page 1: Business Case & Data Presentation
if page == "Business Case & Data Presentation":

    st.header("Business Case")

    st.info("This app predicts student final grades using linear regression.")

    st.subheader("Problem Statement")

    st.write("""
    Student performance is an important issue for schools, teachers, students, and families.
    Many students may struggle academically, but it is not always easy to identify the factors
    that are related to final grades. This app uses student data and linear regression to predict
    a student's final grade based on academic and personal variables.
    """)

    st.subheader("Why This Problem Matters")

    st.write("""
    Predicting student performance can help schools better understand which factors are linked
    to academic outcomes. For example, if study time, previous grades, past failures, or absences
    are related to final grades, educators may be able to identify students who need support earlier.
    The goal of this app is not to label students, but to use data to support better decision-making.
    """)

    st.subheader("App Objective")

    st.write("""
    The objective of this app is to:

    1. Present the student performance dataset.
    2. Visualize important patterns in the data.
    3. Build a linear regression model that predicts a student's final grade.
    4. Allow users to enter student information and receive a predicted final grade.
    """)

    st.header("Dataset Presentation")

    st.write("""
    The dataset contains information about students, including demographic background, family information,
    study time, number of past class failures, absences, and previous grades. The target variable is G3,
    which represents the student's final grade.
    """)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Size")
    st.write(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.")

    st.subheader("Column Names")
    st.write(df.columns.tolist())

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())


# Page 2: Data Visualization

elif page == "Data Visualization":

    st.header("Data Visualization")

    st.write("""
    This page explores important patterns in the student performance dataset. 
    The purpose of these visualizations is to understand which variables may be useful for predicting 
    students' final grades and to identify possible academic risk factors.
    """)

    
    # 1. Distribution of Final Grades
    st.subheader("1. Distribution of Final Grades")

    fig, ax = plt.subplots()
    sns.histplot(df["G3"], kde=True, ax=ax)
    ax.set_xlabel("Final Grade G3")
    ax.set_ylabel("Number of Students")
    ax.set_title("Distribution of Final Grades")
    st.pyplot(fig)

    st.write("""
    The distribution shows that most students received final grades in the middle range, especially 
    around 8 to 15 out of 20. There is also a noticeable group of students with a final grade of 0. 
    This may represent students who failed, dropped the course, or did not complete the final assessment. 
    This pattern matters because these very low scores can affect the model and show that student performance 
    is not evenly distributed.
    """)

    # 2. Relationship Between G2 and G3
    st.subheader("2. Relationship Between Previous Grade G2 and Final Grade G3")

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="G2", y="G3", ax=ax)
    ax.set_xlabel("Second Period Grade G2")
    ax.set_ylabel("Final Grade G3")
    ax.set_title("Relationship Between G2 and G3")
    st.pyplot(fig)

    st.write("""
    This scatterplot shows a very strong positive relationship between G2 and G3. Students who received 
    higher grades in the second period usually also received higher final grades. This suggests that 
    previous academic performance is one of the strongest predictors of final performance. However, there 
    are also some students with reasonable G2 scores but a final grade of 0, which may indicate special 
    cases such as failing to complete the final evaluation.
    """)

    # 3. Relationship Between Absences and Final Grade
    st.subheader("3. Relationship Between Absences and Final Grade")

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="absences", y="G3", ax=ax)
    ax.set_xlabel("Number of Absences")
    ax.set_ylabel("Final Grade G3")
    ax.set_title("Relationship Between Absences and Final Grade")
    st.pyplot(fig)

    st.write("""
    The relationship between absences and final grade is less clear than the relationship between G2 
    and G3. Most students have relatively few absences, but their grades still vary widely. A few students 
    have very high numbers of absences, and their grades tend to be lower or moderate. This suggests that 
    absences may matter, but they are probably not enough by themselves to explain final performance.
    """)

    # 4. Correlation Heatmap
    st.subheader("4. Correlation Heatmap")

    numeric_df = df.select_dtypes(include=["int64", "float64"])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, ax=ax)
    ax.set_title("Correlation Heatmap of Numeric Variables")
    st.pyplot(fig)

    st.write("""
    The heatmap confirms that G1 and G2 have the strongest positive correlations with G3. This means 
    students' earlier grades are highly connected to their final grades. The variable failures has a 
    negative relationship with G3, meaning students with more past failures tend to have lower final grades. 
    Other variables, such as study time and absences, show weaker relationships with G3, so they may still 
    contribute to the model but are not the main driving factors.
    """)

    # 5. Final Grade by Number of Past Failures
    st.subheader("5. Final Grade by Number of Past Failures")

    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="failures", y="G3", ax=ax)
    ax.set_xlabel("Number of Past Class Failures")
    ax.set_ylabel("Final Grade G3")
    ax.set_title("Final Grade by Number of Past Failures")
    st.pyplot(fig)

    st.write("""
    This boxplot shows a clear negative pattern between past failures and final grade. 
    Students with no past failures have the highest median final grade, while students with one, two, 
    or three past failures generally have lower median grades. The spread is also quite wide for students 
    with past failures, which means their outcomes vary, but the overall pattern still suggests that past 
    academic difficulty is strongly connected to lower final performance. This makes failures a useful 
    risk-related variable for the prediction model.
    """)

    # 6. Final Grade by Study Time
    st.subheader("6. Final Grade by Study Time")

    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="studytime", y="G3", ax=ax)
    ax.set_xlabel("Study Time Level")
    ax.set_ylabel("Final Grade G3")
    ax.set_title("Final Grade by Study Time")
    st.pyplot(fig)

    st.write("""
    This boxplot shows that students with higher study time levels tend to have slightly higher median 
    final grades. Students in study time levels 3 and 4 have higher median grades than students in levels 
    1 and 2. However, the relationship is not perfectly linear because the boxes overlap across all groups. 
    This means study time may help explain final grades, but it is not as strong of a predictor as previous 
    grades such as G1 and G2. Study time still matters because it reflects students' learning habits, but 
    it should be understood together with other academic factors.
    """)


# Page 3: Prediction Model
elif page == "Prediction Model":

    st.header("Linear Regression Prediction Model")

    st.write("""
    This page uses a linear regression model to predict a student's final grade, G3.
    The model uses selected variables from the dataset as predictors.
    """)

    st.subheader("Model Setup")

    target = "G3"

    features = ["G1", "G2", "studytime", "failures", "absences"]

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write("Target variable:", target)
    st.write("Predictor variables:", features)

    st.subheader("Model Performance")

    st.write(f"Mean Absolute Error: {mae:.2f}")
    st.write(f"R² Score: {r2:.2f}")

    st.write("""
    The Mean Absolute Error shows the average size of the prediction error.
    The R² score shows how much of the variation in final grades can be explained by the model.
    A higher R² score means the model explains more of the outcome.
    """)

    # Actual vs Predicted Plot
    st.subheader("Actual vs Predicted Final Grades")

    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_xlabel("Actual Final Grade G3")
    ax.set_ylabel("Predicted Final Grade G3")
    ax.set_title("Actual vs Predicted Final Grades")
    st.pyplot(fig)

    st.write("""
    This plot shows that the model predicts final grades fairly well for most students. 
    The points generally follow an upward pattern, meaning that students with higher actual final grades 
    also tend to receive higher predicted grades. This makes sense because the model uses G1 and G2, which 
    are strongly related to G3. However, the model has more difficulty with students whose actual final 
    grade is 0. For several of these students, the model still predicts a grade around 6 to 8, which suggests 
    that the model does not fully capture special cases such as failing, dropping the course, or missing the 
    final assessment. Overall, the model is useful for predicting general grade patterns, but it is less 
    accurate for extreme low outcomes.
    """)

    # Model Coefficients
    st.subheader("Model Coefficients")

    coefficients = pd.DataFrame({
        "Variable": features,
        "Coefficient": model.coef_
    })

    st.dataframe(coefficients)

    st.write("""
    The coefficient table shows that G2 is the strongest positive predictor in the model. 
    Its coefficient is much larger than the coefficient for G1, which means the second period grade 
    contributes more strongly to the predicted final grade when the other variables are held constant. 
    Past failures has a negative coefficient, meaning students with more previous class failures are 
    predicted to have lower final grades. Study time also has a small negative coefficient in this model, 
    but this does not necessarily mean studying is harmful. It may be because study time overlaps with 
    other variables, or because students who struggle more may report studying more. Absences has a very 
    small positive coefficient, so it does not appear to be an important driver of prediction in this model. 
    Overall, the model relies mostly on previous academic performance, especially G2.
    """)

    # User Prediction
    st.subheader("Make a Prediction")

    st.write("""
    Enter student information below to predict the final grade.
    """)

    G1_input = st.number_input(
        "First Period Grade G1",
        min_value=0,
        max_value=20,
        value=10
    )

    G2_input = st.number_input(
        "Second Period Grade G2",
        min_value=0,
        max_value=20,
        value=10
    )

    studytime_input = st.number_input(
        "Study Time Level",
        min_value=1,
        max_value=4,
        value=2
    )

    failures_input = st.number_input(
        "Number of Past Class Failures",
        min_value=0,
        max_value=4,
        value=0
    )

    absences_input = st.number_input(
        "Number of Absences",
        min_value=0,
        max_value=100,
        value=5
    )

    user_data = pd.DataFrame({
        "G1": [G1_input],
        "G2": [G2_input],
        "studytime": [studytime_input],
        "failures": [failures_input],
        "absences": [absences_input]
    })

    if st.button("Predict Final Grade"):

        prediction = model.predict(user_data)
        predicted_grade = prediction[0]

        st.success(f"The predicted final grade is: {predicted_grade:.2f}")

        if predicted_grade >= 15:
            st.write("This is a strong predicted performance.")
        elif predicted_grade >= 10:
            st.write("This is a moderate predicted performance.")
        else:
            st.write("This predicted grade may suggest that the student could benefit from extra support.")

        st.write("""
        This prediction gives an estimated final grade based on the information entered.
        It can help users understand how previous grades, study time, failures, and absences may relate
        to academic performance.
        """)