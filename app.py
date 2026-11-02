import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Set page configuration
st.set_page_config(
    page_title="Sleep Disorder Risk Predictor",
    page_icon="🌙",
    layout="centered"
)

st.title("🌙 Sleep Disorder Risk Prediction App")
st.write("""
This application uses an optimized **XGBoost Classifier with SMOTE** to predict an individual's 
sleep disorder risk tier (**Healthy, Mild, Moderate, or Severe**) based on physiological, 
environmental, and lifestyle parameters.
""")

# Helper to load and train model on-the-fly if a pre-saved pickle isn't present
@st.cache_resource
def load_and_train_pipeline():
    # Load raw data
    df = pd.read_csv("sleep_health_dataset.csv")
    
    # Clean sleep duration column range
    df.loc[(df['sleep_duration_hrs'] < 2.0) | (df['sleep_duration_hrs'] > 20.0), 'sleep_duration_hrs'] = np.nan
    
    target = 'sleep_disorder_risk'
    X = df.drop(columns=['person_id', target])
    y = df[target]
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    no_impute_num_cols = ['sleep_duration_hrs']
    impute_num_cols = [
        'age', 'bmi', 'sleep_quality_score',
        'rem_percentage', 'deep_sleep_percentage', 'sleep_latency_mins',
        'wake_episodes_per_night', 'caffeine_mg_before_bed', 'alcohol_units_before_bed',
        'screen_time_before_bed_mins', 'exercise_day', 'steps_that_day',
        'nap_duration_mins', 'stress_score', 'work_hours_that_day',
        'heart_rate_resting_bpm', 'room_temperature_celsius',
        'weekend_sleep_diff_hrs', 'cognitive_performance_score'
    ]
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num_impute', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), impute_num_cols),
            ('num_no_impute', Pipeline([('scaler', StandardScaler())]), no_impute_num_cols),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
        ]
    )
    
    # Optimal model pipeline matching your benchmark experiments
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', XGBClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            random_state=42, 
            eval_metric='mlogloss',
            n_jobs=-1
        ))
    ])
    
    pipeline.fit(X, y_encoded)
    return pipeline, label_encoder

try:
    pipeline, encoder = load_and_train_pipeline()
    st.success("⚡ Optimized XGBoost Model Pipeline loaded successfully!")
except Exception as e:
    st.error(f"Could not initialize the predictive model pipeline: {e}")
    st.stop()

# User Inputs Sidebar / Layout
st.header("User Attributes & Biometrics")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=34)
    gender = st.selectbox("Gender", ["Male", "Female"])
    occupation = st.selectbox("Occupation", ["Driver", "Software Engineer", "Nurse", "Student", "Lawyer", "Doctor", "Teacher", "Salesperson", "Manager", "Accountant", "Engineer"])
    country = st.selectbox("Country", ["USA", "Canada", "UK", "India", "Japan", "Spain", "Germany", "Australia"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    sleep_duration_hrs = st.slider("Sleep Duration (Hours)", min_value=2.0, max_value=18.0, value=7.0)
    sleep_quality_score = st.slider("Sleep Quality Score (1-10)", min_value=1.0, max_value=10.0, value=6.5)
    rem_percentage = st.slider("REM Sleep Percentage", min_value=0.0, max_value=100.0, value=20.0)
    deep_sleep_percentage = st.slider("Deep Sleep Percentage", min_value=0.0, max_value=100.0, value=20.0)
    sleep_latency_mins = st.number_input("Sleep Latency (Minutes)", min_value=0, max_value=180, value=20)

with col2:
    wake_episodes_per_night = st.number_input("Wake Episodes per Night", min_value=0, max_value=20, value=2)
    caffeine_mg_before_bed = st.number_input("Caffeine Before Bed (mg)", min_value=0, max_value=1000, value=50)
    alcohol_units_before_bed = st.number_input("Alcohol Before Bed (Units)", min_value=0.0, max_value=20.0, value=0.0)
    screen_time_before_bed_mins = st.number_input("Screen Time Before Bed (Minutes)", min_value=0, max_value=480, value=60)
    exercise_day = st.selectbox("Exercised Today?", ["Yes", "No"])
    exercise_val = 1 if exercise_day == "Yes" else 0
    steps_that_day = st.number_input("Steps Today", min_value=0, max_value=100000, value=8000)
    nap_duration_mins = st.number_input("Daytime Nap Duration (Minutes)", min_value=0, max_value=480, value=0)
    stress_score = st.slider("Stress Score (1-10)", min_value=1.0, max_value=10.0, value=5.0)
    work_hours_that_day = st.number_input("Work Hours Today", min_value=0.0, max_value=24.0, value=8.0)
    chronotype = st.selectbox("Chronotype", ["Morning", "Neutral", "Evening"])

# Bottom parameters
mental_health_condition = st.selectbox("Mental Health Condition", ["None", "Anxiety", "Depression", "Both"])
heart_rate_resting_bpm = st.number_input("Resting Heart Rate (BPM)", min_value=30, max_value=200, value=65)
sleep_aid_used = st.selectbox("Used Sleep Aid?", ["No", "Yes"])
sleep_aid_val = 1 if sleep_aid_used == "Yes" else 0
shift_work = st.selectbox("Is Shift Work?", ["No", "Yes"])
shift_work_val = 1 if shift_work == "Yes" else 0
room_temperature_celsius = st.slider("Bedroom Temperature (°C)", min_value=10.0, max_value=35.0, value=20.0)
weekend_sleep_diff_hrs = st.number_input("Weekend Sleep Difference (Hours)", min_value=-5.0, max_value=10.0, value=1.0)
season = st.selectbox("Current Season", ["Spring", "Summer", "Autumn", "Winter"])
day_type = st.selectbox("Day Type", ["Weekday", "Weekend"])
cognitive_performance_score = st.slider("Cognitive Performance Score (0-100)", min_value=0.0, max_value=100.0, value=75.0)
felt_rested = st.selectbox("Felt Rested Upon Waking?", ["No", "Yes"])
felt_rested_val = 1 if felt_rested == "Yes" else 0

# Make prediction
if st.button("🔮 Predict Sleep Disorder Risk Tier", use_container_width=True):
    input_dict = {
        'age': age, 'gender': gender, 'occupation': occupation, 'bmi': bmi, 'country': country,
        'sleep_duration_hrs': sleep_duration_hrs, 'sleep_quality_score': sleep_quality_score,
        'rem_percentage': rem_percentage, 'deep_sleep_percentage': deep_sleep_percentage,
        'sleep_latency_mins': sleep_latency_mins, 'wake_episodes_per_night': wake_episodes_per_night,
        'caffeine_mg_before_bed': caffeine_mg_before_bed, 'alcohol_units_before_bed': alcohol_units_before_bed,
        'screen_time_before_bed_mins': screen_time_before_bed_mins, 'exercise_day': exercise_val,
        'steps_that_day': steps_that_day, 'nap_duration_mins': nap_duration_mins, 'stress_score': stress_score,
        'work_hours_that_day': work_hours_that_day, 'chronotype': chronotype, 
        'mental_health_condition': mental_health_condition, 'heart_rate_resting_bpm': heart_rate_resting_bpm,
        'sleep_aid_used': sleep_aid_val, 'shift_work': shift_work_val, 'room_temperature_celsius': room_temperature_celsius,
        'weekend_sleep_diff_hrs': weekend_sleep_diff_hrs, 'season': season, 'day_type': day_type,
        'cognitive_performance_score': cognitive_performance_score, 'felt_rested': felt_rested_val
    }
    
    input_df = pd.DataFrame([input_dict])
    pred_code = pipeline.predict(input_df)[0]
    class_result = encoder.inverse_transform([pred_code])[0]
    
    # Style results
    st.markdown("--- ")
    if class_result == 'Healthy':
        st.success(f"### Prediction Result: **{class_result}** (Low Risk) 🟩")
    elif class_result == 'Mild':
        st.info(f"### Prediction Result: **{class_result}** Risk 🟨")
    elif class_result == 'Moderate':
        st.warning(f"### Prediction Result: **{class_result}** Risk 🟧")
    else:
        st.error(f"### Prediction Result: **{class_result}** Risk 🟥")
