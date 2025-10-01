# 🏥 Breast Cancer Diagnosis Predictor

A professional Streamlit web application that uses Machine Learning to predict breast cancer diagnosis based on cell nucleus measurements from fine needle aspirate (FNA) images.

## 🌟 Features

- **Interactive GUI** with 30 input fields organized in tabs (Mean, Standard Error, Worst values)
- **Real-time Predictions** using Neural Network Classifier
- **Visual Analytics** with probability distribution charts
- **Feature Importance** visualization showing top contributing factors
- **Professional Design** with custom CSS styling and responsive layout
- **Model Persistence** - automatically trains and saves model for faster subsequent runs

## 📋 Prerequisites

- Python 3.8 or higher
- CSV file named `breast-cancer - Copy.csv` in the same directory

## 🚀 Installation

### 1. Navigate to the project directory

```cmd
cd "c:\Users\natha\OneDrive\Documents\Project2 lang gud\CST9"
```

### 2. Activate your virtual environment

```cmd
.venv\Scripts\activate
```

### 3. Install required packages

```cmd
pip install -r requirements.txt
```

## 💻 Running the Application

### Local Development

```cmd
streamlit run BreastCancerClassify_new.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

## 📊 Input Features

The application requires 30 measurements organized into three categories:

### Mean Values (10 features)

- Radius, Texture, Perimeter, Area
- Smoothness, Compactness, Concavity
- Concave Points, Symmetry, Fractal Dimension

### Standard Error (10 features)

- Standard error for each mean value

### Worst Values (10 features)

- Mean of the three largest values for each feature

## 🎯 Model Details

- **Algorithm**: Random Forest Classifier
- **Features**: 30 numerical measurements
- **Output Classes**:
  - **B** (Benign) - Non-cancerous
  - **M** (Malignant) - Cancerous
- **Training**: Automatically trained on first run using the CSV dataset
- **Model Storage**: Saved as `model.pkl` and `scaler.pkl`

## 📦 Deploying to Streamlit Cloud

### 1. Push your code to GitHub

```cmd
git init
git add .
git commit -m "Initial commit"
git remote add origin YOUR_GITHUB_REPO_URL
git push -u origin main
```

### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository, branch, and main file (`BreastCancerClassify_new.py`)
5. Click "Deploy"

### Important Files for Deployment

- `BreastCancerClassify_new.py` - Main application file
- `requirements.txt` - Python dependencies
- `breast-cancer - Copy.csv` - Dataset (must be in the same directory)

## 📁 File Structure

```
CST9/
├── BreastCancerClassify_new.py    # Main Streamlit application
├── breast-cancer - Copy.csv        # Dataset
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── model.pkl                       # Trained model (generated on first run)
└── scaler.pkl                      # Data scaler (generated on first run)
```

## ⚠️ Medical Disclaimer

This application is for educational and demonstration purposes only. The predictions should **NOT** be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

## 🛠️ Technologies Used

- **Streamlit** - Web application framework
- **Scikit-learn** - Machine learning library
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Plotly** - Interactive visualizations

## 📝 License

This project is for educational purposes.

## 👨‍💻 Author

Created with ❤️ using Streamlit and Machine Learning
