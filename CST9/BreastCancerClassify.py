import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Diagnosis Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stApp {
        max-width: 100%;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
        border: none;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #764ba2;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #667eea;
        border-radius: 10px 10px 0px 0px;
        color: white !important;
    }
    /* Make all tab text white */
    .stTabs [data-baseweb="tab-list"] button {
        color: white !important;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: white !important;
        background-color: #764ba2;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="false"] {
        color: white !important;
        opacity: 0.8;
    }
    .stTabs [data-baseweb="tab-list"] button:hover {
        color: white !important;
        opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🏥 Breast Cancer Diagnosis Predictor")
st.markdown("""
    <div style='text-align: center; padding: 10px; background-color: white; border-radius: 10px; margin-bottom: 20px;'>
        <p style='font-size: 18px; color: #555;'>
            This application uses machine learning to predict breast cancer diagnosis based on cell nucleus measurements.
            <br><strong>Enter the measurements below to get a prediction.</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Load and train model
@st.cache_resource
def load_model():
    """Load the pre-trained model"""
    with open('neural_network_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler


# Load the model
model, scaler = load_model()

# Check if model loaded successfully
if model is None or scaler is None:
    st.stop()  # Stop execution if model files are missing

# File Upload Section
st.markdown("### 📁 Input Method")
input_method = st.radio(
    "Choose how to input data:",
    ["Manual Entry", "Upload CSV File"],
    horizontal=True
)

if input_method == "Upload CSV File":
    st.markdown("#### Upload Patient Data CSV")
    st.info("""
        **CSV Format Requirements:**
        - **Optional:** First column can be 'id' to track patients (will be included in results)
        - Must contain 30 feature columns in EXACT order (after 'id' if present)
        - **Column order:** id (optional), radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, 
          compactness_mean, concavity_mean, concave points_mean, symmetry_mean, fractal_dimension_mean,
          radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, 
          concave points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, 
          perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, 
          concave points_worst, symmetry_worst, fractal_dimension_worst
        - **Important:** "concave points" has a SPACE, not underscore
        - Can contain multiple rows for batch prediction
        
        **Example:** id,radius_mean,texture_mean,perimeter_mean,...
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read the uploaded CSV
            upload_df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! Found {len(upload_df)} row(s) of data.")
            
            # Show preview
            with st.expander("📊 Preview Uploaded Data"):
                st.dataframe(upload_df)
            
            # Batch prediction button
            if st.button("🔬 PREDICT ALL ROWS", use_container_width=True):
                if model is not None and scaler is not None:
                    try:
                        # Validate columns - MUST be in exact order (with spaces in "concave points")
                        expected_cols = [
                            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                            'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
                            'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
                            'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
                            'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
                            'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
                            'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
                        ]
                        
                        # Check if all columns exist
                        missing_cols = [col for col in expected_cols if col not in upload_df.columns]
                        if missing_cols:
                            st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
                        else:
                            # CRITICAL: Check if columns are in the EXACT order
                            # Handle optional 'id' column at the beginning
                            csv_cols = upload_df.columns.tolist()
                            
                            # Check if first column is 'id' (optional)
                            has_id_column = csv_cols[0].lower() == 'id' if len(csv_cols) > 0 else False
                            
                            # Extract feature columns (skip 'id' if present)
                            if has_id_column:
                                feature_cols = csv_cols[1:31]  # Get 30 feature columns after 'id'
                            else:
                                feature_cols = csv_cols[:30]  # Get first 30 columns
                            
                            # Check if the order matches exactly
                            if feature_cols != expected_cols:
                                st.error("❌ **Column Order Error**: The CSV columns must be in the EXACT order specified!")
                                if has_id_column:
                                    st.info("✓ 'id' column detected at the beginning (optional)")
                                st.error("**Expected Order (after 'id' if present):**")
                                st.code(", ".join(expected_cols), language="text")
                                st.error("**Your CSV Order (feature columns):**")
                                st.code(", ".join(feature_cols), language="text")
                                
                                # Show which columns are out of order
                                mismatches = []
                                for i, (expected, actual) in enumerate(zip(expected_cols, feature_cols)):
                                    if expected != actual:
                                        mismatches.append(f"Position {i+1}: Expected '{expected}', but found '{actual}'")
                                
                                if mismatches:
                                    st.error("**Mismatched Positions:**")
                                    for mismatch in mismatches[:5]:  # Show first 5 mismatches
                                        st.write(f"- {mismatch}")
                                    if len(mismatches) > 5:
                                        st.write(f"... and {len(mismatches) - 5} more mismatches")
                            else:
                                # Scale and predict
                                X_upload = upload_df[expected_cols]
                                X_scaled = scaler.transform(X_upload)
                                predictions = model.predict(X_scaled)
                                probabilities = model.predict_proba(X_scaled)
                                
                                # Create results dataframe
                                results_df = upload_df.copy()
                                results_df['Prediction'] = predictions
                                results_df['Benign_Probability'] = probabilities[:, 0]
                                results_df['Malignant_Probability'] = probabilities[:, 1]
                                results_df['Confidence'] = probabilities.max(axis=1) * 100
                                
                                # Display results
                                st.markdown("---")
                                st.markdown("## 📊 Batch Prediction Results")
                                
                                # Summary statistics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    benign_count = (predictions == 'B').sum()
                                    st.metric("✅ Benign Cases", benign_count)
                                with col2:
                                    malignant_count = (predictions == 'M').sum()
                                    st.metric("⚠️ Malignant Cases", malignant_count)
                                with col3:
                                    avg_confidence = results_df['Confidence'].mean()
                                    st.metric("📈 Avg Confidence", f"{avg_confidence:.2f}%")
                                
                                # Results table
                                st.markdown("### 📋 Detailed Results")
                                # Show ID column if it exists in the uploaded file
                                if 'id' in results_df.columns:
                                    display_cols = ['id', 'Prediction', 'Confidence', 'Benign_Probability', 'Malignant_Probability']
                                else:
                                    display_cols = ['Prediction', 'Confidence', 'Benign_Probability', 'Malignant_Probability']
                                st.dataframe(results_df[display_cols])
                                
                                # Download results
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                label="� Download Results as CSV",
                                data=csv,
                                file_name="prediction_results.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                            
                    except Exception as e:
                        st.error(f"❌ Error during batch prediction: {str(e)}")
                else:
                    st.error("❌ Model not loaded. Please check if the dataset file exists.")
                    
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            st.info("Please make sure your CSV file is properly formatted.")
else:
    # Manual Entry (existing functionality)
    st.markdown("### Enter Cell Nucleus Measurements Manually")

# Create tabs for different measurement categories (only show for manual entry)
if input_method == "Manual Entry":
    tab1, tab2, tab3 = st.tabs(["🔵 Mean Values", "📏 Standard Error", "🔴 Worst Values"])
    
    with tab1:
        st.markdown("#### Mean Values of Cell Nucleus Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            radius_mean = st.number_input("Radius Mean", value=0.0, step=0.00001, format="%.5f",
                                         help="Mean of distances from center to points on the perimeter")
            texture_mean = st.number_input("Texture Mean", value=0.0, step=0.00001, format="%.5f",
                                          help="Standard deviation of gray-scale values")
            perimeter_mean = st.number_input("Perimeter Mean", value=0.0, step=0.00001, format="%.5f")
            area_mean = st.number_input("Area Mean", value=0.0, step=0.00001, format="%.5f")
        
        with col2:
            smoothness_mean = st.number_input("Smoothness Mean", value=0.0, step=0.00001, format="%.5f",
                                             help="Local variation in radius lengths")
            compactness_mean = st.number_input("Compactness Mean", value=0.0, step=0.00001, format="%.5f",
                                              help="Perimeter² / area - 1.0")
            concavity_mean = st.number_input("Concavity Mean", value=0.0, step=0.00001, format="%.5f",
                                            help="Severity of concave portions of the contour")
            concave_points_mean = st.number_input("Concave Points Mean", value=0.0, step=0.00001, format="%.5f",
                                                 help="Number of concave portions of the contour")
        
        with col3:
            symmetry_mean = st.number_input("Symmetry Mean", value=0.0, step=0.00001, format="%.5f")
            fractal_dimension_mean = st.number_input("Fractal Dimension Mean", value=0.0, step=0.00001, format="%.5f",
                                                    help="Coastline approximation - 1")

    with tab2:
        st.markdown("#### Standard Error of Cell Nucleus Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            radius_se = st.number_input("Radius SE", value=0.0, step=0.00001, format="%.5f")
            texture_se = st.number_input("Texture SE", value=0.0, step=0.00001, format="%.5f")
            perimeter_se = st.number_input("Perimeter SE", value=0.0, step=0.00001, format="%.5f")
            area_se = st.number_input("Area SE", value=0.0, step=0.00001, format="%.5f")
        
        with col2:
            smoothness_se = st.number_input("Smoothness SE", value=0.0, step=0.00001, format="%.5f")
            compactness_se = st.number_input("Compactness SE", value=0.0, step=0.00001, format="%.5f")
            concavity_se = st.number_input("Concavity SE", value=0.0, step=0.00001, format="%.5f")
            concave_points_se = st.number_input("Concave Points SE", value=0.0, step=0.00001, format="%.5f")

        with col3:
            symmetry_se = st.number_input("Symmetry SE", value=0.0, step=0.00001, format="%.5f")
            fractal_dimension_se = st.number_input("Fractal Dimension SE", value=0.0, step=0.00001, format="%.5f")

    with tab3:
        st.markdown("#### Worst (Largest) Values of Cell Nucleus Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            radius_worst = st.number_input("Radius Worst", value=0.0, step=0.00001, format="%.5f")
            texture_worst = st.number_input("Texture Worst", value=0.0, step=0.00001, format="%.5f")
            perimeter_worst = st.number_input("Perimeter Worst", value=0.0, step=0.00001, format="%.5f")
            area_worst = st.number_input("Area Worst", value=0.0, step=0.00001, format="%.5f")

        with col2:
            smoothness_worst = st.number_input("Smoothness Worst", value=0.0, step=0.00001, format="%.5f")
            compactness_worst = st.number_input("Compactness Worst", value=0.0, step=0.00001, format="%.5f")
            concavity_worst = st.number_input("Concavity Worst", value=0.0, step=0.00001, format="%.5f")
            concave_points_worst = st.number_input("Concave Points Worst", value=0.0, step=0.00001, format="%.5f")
        
        with col3:
            symmetry_worst = st.number_input("Symmetry Worst", value=0.0, step=0.00001, format="%.5f")
            fractal_dimension_worst = st.number_input("Fractal Dimension Worst", value=0.0, step=0.00001, format="%.5f")

    # Create prediction button (only for manual entry)
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔬 PREDICT DIAGNOSIS", use_container_width=True)

    # Make prediction
    if predict_button:
        # Initialize error flag
        has_error = False
        error_messages = []
        
        # Error Handling: Check if model and scaler are loaded
        if model is None or scaler is None:
            st.error("❌ Model could not be loaded. Please ensure the 'breast-cancer - Copy.csv' file exists in the same directory.")
            has_error = True
        
        if not has_error:
            # Collect all input values
            try:
                input_values = [
                    radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, 
                    compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, 
                    fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se,
                    smoothness_se, compactness_se, concavity_se, concave_points_se,
                    symmetry_se, fractal_dimension_se, radius_worst, texture_worst,
                    perimeter_worst, area_worst, smoothness_worst, compactness_worst,
                    concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst
                ]
                
                # Error Handling: Check for None values
                if None in input_values:
                    st.error("❌ Error: Some input fields contain invalid values. Please check all fields.")
                    has_error = True
                
                # Error Handling: Check for NaN or Inf values
                if not has_error:
                    for i, val in enumerate(input_values):
                        if np.isnan(val) or np.isinf(val):
                            st.error("❌ Error: Invalid value detected (NaN or Infinity). Please check your inputs.")
                            has_error = True
                            break
                
                # Error Handling: Check if all values are zero
                if not has_error and all(val == 0.0 for val in input_values):
                    st.warning("⚠️ Warning: All input values are zero. Please enter actual measurement values for accurate prediction.")
                    has_error = True
                
                # Error Handling: Validate reasonable ranges (optional warnings)
                if not has_error:
                    # Check for negative values where they shouldn't exist
                    if any(val < 0 for val in input_values):
                        st.warning("⚠️ Warning: Some values are negative. Please verify your measurements.")
                    
                    # Check for extremely large values (potential data entry errors)
                    if radius_mean > 100 or area_mean > 10000:
                        st.warning("⚠️ Warning: Some values seem unusually large. Please verify your measurements.")
                
                if not has_error:
                    # Create DataFrame with correct column names (matching the training data)
                    # Note: The dataset uses "concave points" with a space, not underscore
                    column_names = [
                        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                        'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
                        'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
                        'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
                        'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
                        'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
                        'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
                    ]
                    input_data = pd.DataFrame([input_values], columns=column_names)
                    
                    # Error Handling: Validate array shape
                    if input_data.shape[1] != 30:
                        st.error(f"❌ Error: Expected 30 features but got {input_data.shape[1]}. Please contact support.")
                        has_error = True
                    
            except ValueError as e:
                st.error(f"❌ Value Error: {str(e)}. Please check that all inputs are valid numbers.")
                has_error = True
            except TypeError as e:
                st.error(f"❌ Type Error: {str(e)}. Please ensure all fields contain numerical values.")
                has_error = True
            except Exception as e:
                st.error(f"❌ Unexpected Error during input collection: {str(e)}")
                has_error = True
        
        # Proceed with prediction if no errors
        if not has_error:
            try:
                # Scale the input
                input_scaled = scaler.transform(input_data)
                
                # Error Handling: Check scaled data validity
                if np.isnan(input_scaled).any() or np.isinf(input_scaled).any():
                    st.error("❌ Error: Data scaling produced invalid values. Please check your inputs.")
                    has_error = True
                
            except ValueError as e:
                st.error(f"❌ Scaling Error: {str(e)}. The input data shape may be incorrect.")
                has_error = True
            except Exception as e:
                st.error(f"❌ Error during data scaling: {str(e)}")
                has_error = True
        
        if not has_error:
            try:
                # Make prediction
                prediction = model.predict(input_scaled)[0]
                prediction_proba = model.predict_proba(input_scaled)[0]
                
                # Error Handling: Validate prediction output
                if prediction not in ['B', 'M']:
                    st.error(f"❌ Error: Invalid prediction result '{prediction}'. Expected 'B' or 'M'.")
                    has_error = True
                
                # Error Handling: Validate probability values
                if len(prediction_proba) != 2:
                    st.error(f"❌ Error: Invalid probability array length. Expected 2, got {len(prediction_proba)}.")
                    has_error = True
                elif not (0 <= prediction_proba[0] <= 1 and 0 <= prediction_proba[1] <= 1):
                    st.error("❌ Error: Probability values are out of valid range [0, 1].")
                    has_error = True
                elif abs(sum(prediction_proba) - 1.0) > 0.01:
                    st.error("❌ Error: Probabilities do not sum to 1.")
                    has_error = True
                    
            except AttributeError as e:
                st.error(f"❌ Model Error: {str(e)}. The model may not be properly trained.")
                has_error = True
            except Exception as e:
                st.error(f"❌ Prediction Error: {str(e)}")
                has_error = True
        
        # Display results only if no errors occurred
        if not has_error and model is not None and scaler is not None:
            try:
                # Display results
                st.markdown("---")
                st.markdown("## 📋 Diagnosis Results")
                
                # Play audio and show animations based on prediction
                import os
                script_dir = os.path.dirname(os.path.abspath(__file__))
                
                if prediction == 'M':
                    # Malignant - play shocked.mp3 and show snow animation
                    st.snow()  # Snow animation for somber effect
                    try:
                        audio_path = os.path.join(script_dir, 'shocked.mp3')
                        audio_file = open(audio_path, 'rb')
                        audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format='audio/mp3', autoplay=True)
                        audio_file.close()
                    except FileNotFoundError:
                        pass  # File not found, skip audio
                else:
                    # Benign - play clapping.mp3 and show balloons animation
                    st.balloons()  # Celebration animation
                    try:
                        audio_path = os.path.join(script_dir, 'clapping.mp3')
                        audio_file = open(audio_path, 'rb')
                        audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format='audio/mp3', autoplay=True)
                        audio_file.close()
                    except FileNotFoundError:
                        pass  # File not found, skip audio
                
                # Create result cards
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction == 'M':
                        # Red text for malignant
                        st.markdown("""
                            <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                                <h3 style='text-align: center; color: #e74c3c;'>Prediction</h3>
                                <h1 style='text-align: center; color: #e74c3c;'>⚠️ MALIGNANT</h1>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        # Default color for benign
                        st.markdown("""
                            <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                                <h3 style='text-align: center; color: #667eea;'>Prediction</h3>
                                <h1 style='text-align: center; color: #27ae60;'>✅ BENIGN</h1>
                            </div>
                            """, unsafe_allow_html=True)
                
                with col2:
                    confidence = max(prediction_proba) * 100
                    st.markdown(f"""
                        <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                            <h3 style='text-align: center; color: #667eea;'>Confidence</h3>
                            <h1 style='text-align: center; color: #3498db;'>{confidence:.2f}%</h1>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col3:
                    # Determine risk level and color
                    if prediction == 'M':
                        if confidence >= 80:
                            risk_level = "HIGH"
                            risk_color = "#e74c3c"  # Red
                            risk_icon = "🔴"
                        else:
                            risk_level = "MODERATE"
                            risk_color = "#f39c12"  # Yellow/Orange
                            risk_icon = "🟡"
                    else:
                        if confidence >= 80:
                            risk_level = "LOW"
                            risk_color = "#27ae60"  # Green
                            risk_icon = "🟢"
                        else:
                            risk_level = "MODERATE"
                            risk_color = "#f39c12"  # Yellow/Orange
                            risk_icon = "🟡"
                    
                    st.markdown(f"""
                        <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                            <h3 style='text-align: center; color: #667eea;'>Risk Level</h3>
                            <h1 style='text-align: center; color: {risk_color};'>{risk_icon} {risk_level}</h1>
                        </div>
                        """, unsafe_allow_html=True)
            
                # Probability chart
                st.markdown("### 📊 Probability Distribution")
                
                # Determine the order for display
                if prediction == 'B':
                    labels = ['Benign (Predicted)', 'Malignant']
                    values = [prediction_proba[0], prediction_proba[1]]
                    colors = ['#27ae60', '#e74c3c']
                else:
                    labels = ['Benign', 'Malignant (Predicted)']
                    values = [prediction_proba[0], prediction_proba[1]]
                    colors = ['#27ae60', '#e74c3c']
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=labels,
                        y=values,
                        marker_color=colors,
                        text=[f'{val*100:.2f}%' for val in values],
                        textposition='auto',
                    )
                ])
                fig.update_layout(
                    title="Prediction Probabilities",
                    yaxis_title="Probability",
                    xaxis_title="Diagnosis",
                    height=400,
                    showlegend=False,
                    yaxis=dict(range=[0, 1])
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance visualization
                st.markdown("### 🎯 Key Features")
                feature_names = [
                    'Radius Mean', 'Texture Mean', 'Perimeter Mean', 'Area Mean', 'Smoothness Mean',
                    'Compactness Mean', 'Concavity Mean', 'Concave Points Mean', 'Symmetry Mean',
                    'Fractal Dimension Mean', 'Radius SE', 'Texture SE', 'Perimeter SE', 'Area SE',
                    'Smoothness SE', 'Compactness SE', 'Concavity SE', 'Concave Points SE',
                    'Symmetry SE', 'Fractal Dimension SE', 'Radius Worst', 'Texture Worst',
                    'Perimeter Worst', 'Area Worst', 'Smoothness Worst', 'Compactness Worst',
                    'Concavity Worst', 'Concave Points Worst', 'Symmetry Worst', 'Fractal Dimension Worst'
                ]
                
                # Get feature importances
                importances = model.feature_importances_
                top_10_idx = np.argsort(importances)[-10:]
                top_10_features = [feature_names[i] for i in top_10_idx]
                top_10_importances = importances[top_10_idx]
            
                fig2 = go.Figure(data=[
                    go.Bar(
                        y=top_10_features,
                        x=top_10_importances,
                        orientation='h',
                        marker_color='#667eea',
                        text=[f'{val:.4f}' for val in top_10_importances],
                        textposition='auto',
                    )
                ])
                fig2.update_layout(
                    title="Top 10 Most Important Features in This Model",
                    xaxis_title="Feature Importance",
                    yaxis_title="Feature",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                # Disclaimer
                st.markdown("""
                    <div style='background-color: yellow; color: black; padding: 15px; border-radius: 10px; border-left: 5px solid #ffc107; margin-top: 20px;'>
                        <strong>⚠️ Medical Disclaimer:</strong> This prediction is based on machine learning and should NOT be used as a substitute 
                        for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified 
                        health provider with any questions you may have regarding a medical condition.
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"❌ Error displaying results: {str(e)}")
                st.error("Please try again or contact support if the problem persists.")

# Sidebar information
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.info("""
        This application predicts whether a breast mass is **benign** or **malignant** based on measurements 
        of cell nuclei present in a digitized image of a fine needle aspirate (FNA) of a breast mass.
    """)
    
    st.markdown("### 📖 Feature Descriptions")
    with st.expander("Mean Values"):
        st.write("""
        - **Radius**: Mean distance from center to perimeter points
        - **Texture**: Standard deviation of gray-scale values
        - **Perimeter**: Perimeter of the cell nucleus
        - **Area**: Area of the cell nucleus
        - **Smoothness**: Local variation in radius lengths
        - **Compactness**: Perimeter² / area - 1.0
        - **Concavity**: Severity of concave portions
        - **Concave Points**: Number of concave portions
        - **Symmetry**: Symmetry of the cell nucleus
        - **Fractal Dimension**: "Coastline approximation" - 1
        """)
    
    with st.expander("Standard Error"):
        st.write("Standard error for each of the above features")
    
    with st.expander("Worst Values"):
        st.write("Mean of the three largest values for each feature")
    
    st.markdown("### 🎯 Model Information")
    st.success("""
        **Algorithm**: Neural Network  
        **Features**: 30 measurements  
        **Classes**: Benign (B) / Malignant (M)  
    """)
    
    st.markdown("### 📊 Dataset")
    st.write("""
        The dataset contains features computed from digitized images of fine needle aspirate (FNA) 
        of breast masses, describing characteristics of the cell nuclei present in the images.
        
        **Classes:**
        - **M** = Malignant (Cancerous)
        - **B** = Benign (Non-cancerous)
    """)
    
    st.markdown("---")
    st.markdown("### 🚀 Quick Start")
    st.write("""
        1. Enter measurements in the tabs above
        2. Click **PREDICT DIAGNOSIS**
        3. View results and probability distribution
    """)
    
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit")

# Copyright footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 20px; background-color: black; border-radius: 10px; margin-top: 30px;'>
        <p style='color: white; font-size: 14px; margin: 0;'>
            © 2025 Breast Cancer Diagnosis Predictor. All Rights Reserved.
        </p>
        <p style='color: white; font-size: 12px; margin-top: 5px;'>
            Developed for educational and research purposes only. Not for clinical use.
        </p>
        <p style='color: white; font-size: 12px; margin-top: 10px;'>
            👤 Developed by: <strong style='color: #667eea;'>Nathaniel Keene Merka, Justin Troy Rosalada, Lloyd Justin Felecilda</strong><br>
            📧 Contact: <a href='mailto:nkmerka.work@gmail.com' style='color: #667eea; text-decoration: none;'>nkmerka.work@gmail.com</a><br>
            💻 GitHub: <a href='https://github.com/Siege365' target='_blank' style='color: #667eea; text-decoration: none;'>github.com/Siege365</a>
        </p>
    </div>
    """, unsafe_allow_html=True)
