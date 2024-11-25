### EEG-Based Epileptic Seizure Prediction Workflow 

This project integrates EEG signal processing, machine learning-based prediction, workflow management, and containerization for automated, reproducible results.  

---

## Project Structure
├── Dockerfile          # Defines the Docker container setup
├── workflow.nf             # Nextflow pipeline definition
├── src.py            # Python script for model training
├── app.py              # Streamlit app for frontend interaction
├── Model.pkl           # Generated model file (output from model.py)
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation

## Model Overview  

### Prediction Task 
- Binary Classification: Predicts **Epileptic (1)** or **Non-Epileptic (0)** based on EEG signals.  

### **Model Type**  
- **Voting Classifier** (Soft Voting): Combines predictions from:
  - XGBoost  
  - K-Nearest Neighbors (KNN)  
  - Decision Tree  
  - Random Forest  
  - Multi-Layer Perceptron (MLP)  

---

## Data Overview  

- **Source:**  
  - [UCI EEG Database](https://doi.org/10.24432/C5TS3D)  
  - [Kaggle EEG Dataset](https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition)  

### **Preprocessing Steps**  
1. **Class Balancing:** SMOTE to handle class imbalance.  
2. **Feature Scaling:** StandardScaler for standardization (mean=0, std=1).  
3. **Signal Processing:** Discrete Wavelet Transform (DWT) for feature extraction.  
4. **Dimensionality Reduction:** PCA (95% variance retained).  

### **Performance Metrics**  
- **Accuracy**, **Precision**, **Recall**, **F1 Score**, and **Confusion Matrix**.
- **Cross-validation**: K-fold (k=5)

### **Model Saving**  
- Trained model saved as `Model.pkl` using **pickle** for future predictions.  

---

## Workflow Automation  

### **Nextflow Workflow**  
- Automates the **model training** (`src.py`) and **Streamlit app** (`app.py`) integration.  
- Ensures reproducibility by integrating backend and frontend seamlessly.  

### **Streamlit App**  
- Provides a user-friendly interface for prediction using the trained model.  
The app can be accessible via: https://my-python-app-5oiwmq3ipalkmzagedwmyy.streamlit.app/
---

## Containerization

### **Docker Integration**  
- Environment is containerized with Docker for reproducibility and consistency.  
- Includes:
  - Nextflow pipeline (`workflow.nf`).  
  - Python dependencies for `src.py` and `app.py`.  
  - Predefined environment (Python, libraries, Nextflow setup al included in requirements.txt file).  

### **Execution**  
1. **Build Docker Image:**  
   ```bash
   sudo docker build -t nextflow-workflow .       #'nextflow-workflow' is my docker-image
2. **Run container:**
   ```bash
   sudo docker run -it -p 8080:8080 nextflow-workflow
   
3. **Web address to find the API:** 
https://my-python-app-5oiwmq3ipalkmzagedwmyy.streamlit.app/

   ## Summary
   This project offers a scalable and interactive solution for predicting epileptic seizures using EEG signals. By combining EEG signal processing and machine learning, the solution is deployed via a Streamlit app for easy access and use. The trained model is available for real-time predictions and can be deployed using Streamlit Cloud for global access.
