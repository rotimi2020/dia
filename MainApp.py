import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import joblib
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Assessment Suite",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for updated styling
st.markdown("""
<style>
    body {
        background-color: #f0f2f6;
        color: #31333F;
    }
    .stApp {
        background: #ffffff;
    }
    .main-header {
        font-size: 2.2rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #FF4B4B;
    }
    .section-header {
        font-size: 1.6rem;
        color: #FF4B4B;
        margin-top: 1.8rem;
        margin-bottom: 1rem;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #ddd;
    }
    .subsection-header {
        font-size: 1.3rem;
        color: #FF4B4B;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .info-box {
        background-color: #FFF5F5;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #FF4B4B;
        font-size: 0.95rem;
        color: #31333F;
    }
    .metric-box {
        background-color: #FFF5F5;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin: 10px 0;
        border: 1px solid #FFCCCB;
        color: #31333F;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #FF4B4B;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666666;
    }
    .stDataFrame {
        background-color: #ffffff;
        color: #31333F;
    }
    .stSlider, .stNumberInput, .stSelectbox, .stRadio {
        font-size: 0.9rem;
        color: #31333F;
    }
    /* Sidebar styling */
    .css-1d391kg, .css-1d391kg p {
        background-color: #ffffff;
        color: #31333F;
    }
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
        color: #FF4B4B;
    }
    /* Hide empty text elements */
    .stMarkdown:has(> div > div > p:empty) {
        display: none;
    }
    /* Prediction app styles */
    .sub-header {
        font-size: 1.2rem;
        color: #2c3e50;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }
    .risk-high {
        background-color: #ffcccc;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ff4b4b;
        margin: 8px 0;
        font-size: 0.9rem;
    }
    .risk-medium {
        background-color: #fff4cc;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 8px 0;
        font-size: 0.9rem;
    }
    .risk-low {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 8px 0;
        font-size: 0.9rem;
    }
    .result-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border: 1px solid #dee2e6;
        font-size: 0.9rem;
    }
    .screening-box {
        border-color: #ffc107;
        background-color: #fff3cd;
    }
    .diagnostic-box {
        border-color: #dc3545;
        background-color: #f8d7da;
    }
    .positive-result {
        color: #dc3545;
        font-weight: bold;
        font-size: 1rem;
    }
    .negative-result {
        color: #28a745;
        font-weight: bold;
        font-size: 1rem;
    }
    .prediction-info-box {
        background-color: #e8f4f8;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 4px solid #17a2b8;
        font-size: 0.9rem;
    }
    .summary-box {
        background-color: #e9ecef;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        border-left: 4px solid #6c757d;
        font-size: 0.9rem;
    }
    .probability-display {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        margin: 15px 0;
        padding: 12px;
        background-color: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Load data for visualization
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('nhanes_analysis.csv')
        return df
    except:
        st.error("Data file not found. Please ensure 'nhanes_analysis.csv' is in the correct directory.")
        return None

# Load model for prediction
@st.cache_resource
def load_model():
    try:
        model = joblib.load('diabetes_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, scaler, feature_names
    except:
        st.error("Model files not found. Please ensure 'diabetes_model.pkl', 'scaler.pkl', and 'feature_names.pkl' are in the correct directory.")
        return None, None, None

# Create custom color palette
red_palette = ["#FF4B4B", "#FF6B6B", "#FF8E8E", "#FFA8A8", "#FFC2C2", "#FFDCDC"]
red_cmap = LinearSegmentedColormap.from_list("red", ["#FFFFFF", "#FFDCDC", "#FFA8A8", "#FF6B6B", "#FF4B4B"])

# Set plot style
plt.style.use('default')
sns.set_palette(red_palette)

# Mapping dictionaries for prediction
Gender_Code = {1: 'Male', 2: 'Female'}
Race_Code = {
    1: 'Mexican American', 
    2: 'Other Hispanic', 
    3: 'Non-Hispanic White',
    4: 'Non-Hispanic Black', 
    6: 'Non-Hispanic Asian', 
    7: 'Other Race'
}
Education_Code_Imputed = {
    1: 'Less than 9th grade', 
    2: '9-11th grade', 
    3: 'High school graduate',
    4: 'Some college or AA degree', 
    5: 'College graduate'
}
Family_Diabetes_Code_Imputed = {1: 'Yes', 2: 'No'}
Risk_Level = {0: 'High Risk', 1: 'Low Risk'}
Obesity_Status = {0: 'Non-Obese', 1: 'Obese', 2: 'Overweight'}

# Set thresholds
SCREENING_THRESHOLD = 0.7  # High recall for community screening
DIAGNOSTIC_THRESHOLD = 0.9  # High precision for hospital diagnosis

# Visualization app
def visualization_app():
    st.markdown('<h1 class="main-header">Diabetes Risk Assessment Data Visualization</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Sidebar filters
    st.sidebar.markdown("## Data Filters")
    
    # Age filter
    min_age, max_age = int(df['Age_Imputed'].min()), int(df['Age_Imputed'].max())
    age_range = st.sidebar.slider("Age Range", min_age, max_age, (min_age, max_age))
    
    # Diabetes status filter
    diabetes_options = list(df['Diabetes_Status'].unique())
    selected_diabetes = st.sidebar.multiselect("Diabetes Status", diabetes_options, diabetes_options)
    
    # Gender filter
    gender_options = list(df['Gender'].unique())
    selected_gender = st.sidebar.multiselect("Gender", gender_options, gender_options)
    
    # Apply filters
    filtered_df = df[
        (df['Age_Imputed'] >= age_range[0]) & 
        (df['Age_Imputed'] <= age_range[1]) &
        (df['Diabetes_Status'].isin(selected_diabetes)) &
        (df['Gender'].isin(selected_gender))
    ]
    
    # Display dataset info
    st.markdown('<h2 class="section-header">Dataset Overview</h2>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''
            <div class="metric-box">
                <div class="metric-value">{filtered_df.shape[0]}</div>
                <div class="metric-label">Total Records</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        diabetes_count = filtered_df[filtered_df['Diabetes_Status'] == 'Diabetes'].shape[0]
        st.markdown(f'''
            <div class="metric-box">
                <div class="metric-value">{diabetes_count}</div>
                <div class="metric-label">Diabetes Cases</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        avg_age = filtered_df['Age_Imputed'].mean()
        st.markdown(f'''
            <div class="metric-box">
                <div class="metric-value">{avg_age:.1f}</div>
                <div class="metric-label">Average Age</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        avg_bmi = filtered_df['BMI_Imputed'].mean()
        st.markdown(f'''
            <div class="metric-box">
                <div class="metric-value">{avg_bmi:.1f}</div>
                <div class="metric-label">Average BMI</div>
            </div>
        ''', unsafe_allow_html=True)
    
    # Show filtered data
    if st.checkbox("Show Filtered Data"):
        st.dataframe(filtered_df)
    
    # Distribution analysis
    st.markdown('<h2 class="section-header">Distribution Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<h3 class="subsection-header">Diabetes Status</h3>', unsafe_allow_html=True)
        diabetes_counts = filtered_df['Diabetes_Status'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 5))
        wedges, texts, autotexts = ax.pie(diabetes_counts.values, labels=diabetes_counts.index, autopct='%1.1f%%', 
                                         colors=red_palette[:len(diabetes_counts)])
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax.set_title('Diabetes Status Distribution')
        st.pyplot(fig)
    
    with col2:
        st.markdown('<h3 class="subsection-header">Risk Level</h3>', unsafe_allow_html=True)
        risk_counts = filtered_df['Risk_Level'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 5))
        bars = ax.bar(risk_counts.index, risk_counts.values, color=red_palette[:len(risk_counts)])
        ax.set_title('Risk Level Distribution')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        st.pyplot(fig)
    
    with col3:
        st.markdown('<h3 class="subsection-header">Obesity Status</h3>', unsafe_allow_html=True)
        obesity_counts = filtered_df['Obesity_Status'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 5))
        bars = ax.bar(obesity_counts.index, obesity_counts.values, color=red_palette[:len(obesity_counts)])
        ax.set_title('Obesity Status Distribution')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        st.pyplot(fig)
    
    # Health metrics comparison
    st.markdown('<h2 class="section-header">Health Metrics by Diabetes Status</h2>', unsafe_allow_html=True)
    
    diabetes_comparison = filtered_df.groupby('Diabetes_Status')[
        ["BMI_Imputed", "Waist_Circumference_Imputed", "Glucose_Imputed", "Triglycerides_Imputed"]
    ].mean().round(2)
    
    st.dataframe(diabetes_comparison.style.background_gradient(cmap=red_cmap))
    
    # BMI Comparison by Gender and Diabetes Status
    st.markdown('<h2 class="section-header">BMI Comparison by Gender and Diabetes Status</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="subsection-header">Males</h3>', unsafe_allow_html=True)
        male_bmi = filtered_df.query("Gender == 'Male'").groupby("Diabetes_Status")["BMI_Imputed"].mean().round(2)
        fig, ax = plt.subplots(figsize=(6, 5))
        bars = ax.bar(male_bmi.index, male_bmi.values, color=red_palette[:len(male_bmi)])
        ax.set_title('Average BMI - Males')
        ax.set_ylabel('BMI')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height}', ha='center', va='bottom')
        st.pyplot(fig)
    
    with col2:
        st.markdown('<h3 class="subsection-header">Females</h3>', unsafe_allow_html=True)
        female_bmi = filtered_df.query("Gender == 'Female'").groupby("Diabetes_Status")["BMI_Imputed"].mean().round(2)
        fig, ax = plt.subplots(figsize=(6, 5))
        bars = ax.bar(female_bmi.index, female_bmi.values, color=red_palette[:len(female_bmi)])
        ax.set_title('Average BMI - Females')
        ax.set_ylabel('BMI')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height}', ha='center', va='bottom')
        st.pyplot(fig)
    
    # Income analysis
    st.markdown('<h2 class="section-header">Income Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="subsection-header">Income by Obesity Status</h3>', unsafe_allow_html=True)
        income_obesity = filtered_df.groupby("Obesity_Status")["Income"].mean().round(2)
        fig, ax = plt.subplots(figsize=(6, 5))
        bars = ax.bar(income_obesity.index, income_obesity.values, color=red_palette[:len(income_obesity)])
        ax.set_title('Average Income by Obesity Status')
        ax.set_ylabel('Income ($)')
        plt.xticks(rotation=45)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'${height:,.0f}', ha='center', va='bottom')
        st.pyplot(fig)
    
    with col2:
        st.markdown('<h3 class="subsection-header">Income by Education Level (Diabetes Patients)</h3>', unsafe_allow_html=True)
        income_education = filtered_df.query("Diabetes_Status == 'Diabetes'").groupby('Education')['Income'].mean()
        fig, ax = plt.subplots(figsize=(6, 5))
        bars = ax.bar(income_education.index, income_education.values, color=red_palette[:len(income_education)])
        ax.set_title('Income by Education (Diabetes Patients)')
        ax.set_ylabel('Income ($)')
        plt.xticks(rotation=45)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'${height:,.0f}', ha='center', va='bottom')
        st.pyplot(fig)
    
    # Detailed visualizations
    st.markdown('<h2 class="section-header">Detailed Visualizations</h2>', unsafe_allow_html=True)
    
    # Age Distribution
    st.markdown('<h3 class="subsection-header">Age Distribution</h3>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(filtered_df['Age_Imputed'], bins=20, kde=True, color=red_palette[0])
    plt.title('Age Distribution (Imputed)', fontsize=14, fontweight='bold')
    plt.xlabel('Age (years)')
    plt.ylabel('Frequency')
    st.pyplot(fig)
    
    # BMI Distribution
    st.markdown('<h3 class="subsection-header">BMI Distribution</h3>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(filtered_df['BMI_Imputed'], bins=20, kde=True, color=red_palette[1])
    plt.title('BMI Distribution (Imputed)', fontsize=14, fontweight='bold')
    plt.xlabel('BMI')
    plt.ylabel('Frequency')
    st.pyplot(fig)
    
    # Diabetes by Gender
    st.markdown('<h3 class="subsection-header">Diabetes Prevalence by Gender</h3>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=filtered_df, x='Gender', hue='Diabetes_Status', palette=red_palette[:2])
    plt.title('Diabetes Prevalence by Gender', fontsize=14, fontweight='bold')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.legend(title='Diabetes Status')
    st.pyplot(fig)
    
    # Diabetes by Race
    st.markdown('<h3 class="subsection-header">Diabetes Prevalence by Race/Ethnicity</h3>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=filtered_df, y='Race', hue='Diabetes_Status', palette=red_palette[:2])
    plt.title('Diabetes Prevalence by Race/Ethnicity', fontsize=14, fontweight='bold')
    plt.xlabel('Count')
    plt.ylabel('Race/Ethnicity')
    plt.legend(title='Diabetes Status')
    st.pyplot(fig)
    
    # Correlation Heatmap
    st.markdown('<h3 class="subsection-header">Feature Correlation Matrix</h3>', unsafe_allow_html=True)
    imputed_numeric_features = [col for col in filtered_df.columns if col.endswith("_Imputed") and col != 'Income_Code_Imputed']
    correlation_matrix = filtered_df[imputed_numeric_features].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap=red_cmap, center=0,
                square=True, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix (Imputed Values)', fontsize=14, fontweight='bold')
    st.pyplot(fig)
    
    # BMI by Diabetes Status
    st.markdown('<h3 class="subsection-header">BMI Distribution by Diabetes Status</h3>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='Diabetes_Status', y='BMI_Imputed', data=filtered_df, palette=red_palette[:2])
    plt.title('BMI Distribution by Diabetes Status', fontsize=14, fontweight='bold')
    plt.xlabel('Diabetes Status')
    plt.ylabel('BMI')
    st.pyplot(fig)

# Prediction app
def prediction_app():
    st.markdown('<h1 class="main-header">Diabetes Risk Assessment</h1>', unsafe_allow_html=True)
    
    # Load model
    model, scaler, feature_names = load_model()
    if model is None:
        return
    
    st.markdown("""
    <div class="prediction-info-box">
        <strong>Clinical Guidance:</strong> This tool uses a dual-threshold approach:
        <ul>
            <li><strong>Screening (70%):</strong> High recall to identify potential cases</li>
            <li><strong>Diagnostic (90%):</strong> High precision to confirm diagnosis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Create input form
    st.markdown('<h2 class="sub-header">Patient Information</h2>', unsafe_allow_html=True)

    # Demographics
    st.markdown("**Demographic Information**")
    age = st.slider("Age", min_value=1, max_value=100, value=45)
    gender = st.radio("Gender", options=list(Gender_Code.values()), horizontal=True)
    race = st.selectbox("Race/Ethnicity", options=list(Race_Code.values()))
    education = st.selectbox("Education Level", options=list(Education_Code_Imputed.values()))
    income = st.number_input("Annual Income ($)", min_value=2500, max_value=100000, value=50000, step=1000)

    # Clinical measurements
    st.markdown("**Clinical Measurements**")
    col1, col2 = st.columns(2)
    with col1:
        bmi = st.slider("BMI", min_value=10.0, max_value=85.0, value=25.0, step=0.1)
        waist_circumference = st.slider("Waist Circumference (cm)", min_value=40.0, max_value=180.0, value=90.0)
        systolic_bp = st.slider("Systolic BP (mmHg)", min_value=65, max_value=230, value=120)
    with col2:
        diastolic_bp = st.slider("Diastolic BP (mmHg)", min_value=0, max_value=130, value=80)
        glucose = st.slider("Glucose Level (mg/dL)", min_value=40, max_value=610, value=100)
        hdl = st.slider("HDL Level (mg/dL)", min_value=10, max_value=125, value=50)
        triglycerides = st.slider("Triglycerides Level (mg/dL)", min_value=10, max_value=4250, value=150)

    # Risk factors
    st.markdown("**Risk Factors**")
    family_diabetes = st.radio("Family History of Diabetes", options=list(Family_Diabetes_Code_Imputed.values()), horizontal=True)
    risk_level = st.selectbox("Clinical Risk Level", options=list(Risk_Level.values()))
    obesity_status = st.selectbox("Obesity Status", options=list(Obesity_Status.values()))

    # Convert categorical inputs back to numerical codes
    gender_code = [k for k, v in Gender_Code.items() if v == gender][0]
    race_code = [k for k, v in Race_Code.items() if v == race][0]
    education_code = [k for k, v in Education_Code_Imputed.items() if v == education][0]
    family_diabetes_code = [k for k, v in Family_Diabetes_Code_Imputed.items() if v == family_diabetes][0]
    risk_level_code = [k for k, v in Risk_Level.items() if v == risk_level][0]
    obesity_status_code = [k for k, v in Obesity_Status.items() if v == obesity_status][0]

    # Create feature vector in the same order as training
    input_data = pd.DataFrame([[
        age, gender_code, race_code, bmi, waist_circumference, systolic_bp, 
        diastolic_bp, glucose, hdl, triglycerides, education_code, 
        family_diabetes_code, income, risk_level_code, obesity_status_code
    ]], columns=feature_names)

    # Scale the input data
    input_scaled = scaler.transform(input_data)

    # Make prediction
    if st.button("Assess Diabetes Risk", type="primary", use_container_width=True):
        probability = model.predict_proba(input_scaled)[0][1]
        probability_percent = probability * 100
        
        st.markdown("---")
        st.markdown('<h2 class="sub-header">Assessment Results</h2>', unsafe_allow_html=True)
        
        # Display the probability in percentage
        st.markdown(f'<div class="probability-display">Diabetes Probability: {probability_percent:.1f}%</div>', unsafe_allow_html=True)
        
        # Screening result
        st.markdown(f'<div class="result-box screening-box">', unsafe_allow_html=True)
        st.markdown("##### Screening Result (High Recall)")
        if probability >= SCREENING_THRESHOLD:
            st.markdown('<p class="positive-result">SCREENING POSITIVE</p>', unsafe_allow_html=True)
            st.markdown(f"*Probability ({probability_percent:.1f}%) ≥ Screening Threshold (70%)*")
            st.markdown("**Recommendation:** Refer for diagnostic testing")
        else:
            st.markdown('<p class="negative-result">SCREENING NEGATIVE</p>', unsafe_allow_html=True)  
            st.markdown(f"*Probability ({probability_percent:.1f}%) < Screening Threshold (70%)*")
            st.markdown("**Recommendation:** No further testing needed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Diagnostic result
        st.markdown(f'<div class="result-box diagnostic-box">', unsafe_allow_html=True)
        st.markdown("##### Diagnostic Result (High Precision)")
        if probability >= DIAGNOSTIC_THRESHOLD:
            st.markdown('<p class="positive-result">DIAGNOSTIC POSITIVE</p>', unsafe_allow_html=True)
            st.markdown(f"*Probability ({probability_percent:.1f}%) ≥ Diagnostic Threshold (90%)*")
            st.markdown("**Recommendation:** High confidence of diabetes")
        else:
            st.markdown('<p class="negative-result">DIAGNOSTIC NEGATIVE</p>', unsafe_allow_html=True)
            st.markdown(f"*Probability ({probability_percent:.1f}%) < Diagnostic Threshold (90%)*")
            st.markdown("**Recommendation:** Insufficient evidence for diagnosis")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Summary
        st.markdown('<div class="summary-box">', unsafe_allow_html=True)
        st.markdown("##### Clinical Summary")
        
        if probability >= DIAGNOSTIC_THRESHOLD:
            st.markdown("**HIGH CONFIDENCE OF DIABETES**")
            st.markdown("Immediate confirmatory testing and treatment planning recommended.")
        elif probability >= SCREENING_THRESHOLD:
            st.markdown("**ELEVATED RISK OF DIABETES**")
            st.markdown("Further diagnostic testing advised.")
        else:
            st.markdown("**LOW RISK OF DIABETES**")
            st.markdown("Continue routine preventive care.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Risk factors visualization
        st.markdown("---")
        st.markdown('<h3 class="sub-header">Risk Factor Analysis</h3>', unsafe_allow_html=True)
        
        # Create a simple bar chart of the most influential features
        feature_importance = model.feature_importances_
        indices = np.argsort(feature_importance)[::-1]
        
        # Get top 8 features
        top_features = [feature_names[i] for i in indices[:8]]
        top_importance = [feature_importance[i] for i in indices[:8]]
        
        # Create a more readable feature name mapping
        feature_name_map = {
            'Age': 'Age',
            'Gender_Code': 'Gender',
            'Race_Code': 'Race/Ethnicity',
            'BMI_Imputed': 'BMI',
            'Waist_Circumference_Imputed': 'Waist Circumference',
            'Systolic_BP_Imputed': 'Systolic BP',
            'Diastolic_BP_Imputed': 'Diastolic BP',
            'Glucose_Imputed': 'Glucose Level',
            'HDL_Imputed': 'HDL Level',
            'Triglycerides_Imputed': 'Triglycerides',
            'Education_Code_Imputed': 'Education Level',
            'Family_Diabetes_Code_Imputed': 'Family History',
            'Income': 'Income',
            'Risk_Level': 'Clinical Risk Level',
            'Obesity_Status': 'Obesity Status'
        }
        
        readable_features = [feature_name_map.get(f, f) for f in top_features]
        
        fig = go.Figure(go.Bar(
            x=top_importance,
            y=readable_features,
            orientation='h',
            marker_color='#3498db'
        ))
        
        fig.update_layout(
            title="Top Influential Risk Factors",
            xaxis_title="Importance",
            yaxis_title="Factor",
            height=300,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="prediction-info-box">
    <strong>Disclaimer:</strong> This tool is for clinical decision support only and should not replace professional medical judgment. 
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center;">
        <h3>Diabetes Risk Assessor</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### Threshold Information")
        
        st.markdown("""
        <div class="result-box screening-box">
        <h4>Screening Threshold</h4>
        <h3>70%</h3>
        <p>High recall for community screening</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="result-box diagnostic-box">
        <h4>Diagnostic Threshold</h4>
        <h3>90%</h3>
        <p>High precision for clinical diagnosis</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### Interpretation Guide")
        
        st.markdown("""
        - **< 70%:** Screening Negative
        - **70-89%:** Screening Positive
        - **≥ 90%:** Diagnostic Positive
        """)
        
        st.markdown("---")
        st.caption("Clinical Decision Support Tool")

# Main app
def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Select Mode", ["Data Visualization", "Risk Assessment"])
    
    if app_mode == "Data Visualization":
        visualization_app()
    else:
        prediction_app()

if __name__ == "__main__":
    main()