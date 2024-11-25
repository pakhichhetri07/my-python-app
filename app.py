import os
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Set Streamlit page layout and title
st.set_page_config(page_title="Epileptic Seizure Prediction App", layout="wide", initial_sidebar_state="expanded")

# Add custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #f5f5f5; /* Soft grey background */
            color: #333; /* Default text color */
        }
        .stSidebar {
            background-color: #f0f0f0; /* Light grey sidebar */
        }
        .main-title {
            font-size: 2.5em;
            color: #4CAF50; /* Green color for main title */
            text-align: center;
        }
        .sub-title {
            font-size: 1.2em;
            color: #555; /* Subtle dark grey */
            text-align: center;
        }
        .section-header {
            font-size: 1.8em;
            color: #2196F3; /* Blue section headers */
            margin-top: 1em;
        }
        .footer {
            text-align: center;
            font-size: 0.9em;
            color: #888;
        }
    </style>
""", unsafe_allow_html=True)

# Load the pre-trained model
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(".", "Model.pkl")  # Model directory
        if not os.path.exists(model_path):
            st.error(f"Model.pkl not found in: {model_path}")
            return None

        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Function to make predictions
def make_prediction(features):
    try:
        input_data = np.array(features, dtype=np.float64)

        if np.isnan(input_data).any() or any(f is None for f in features):
            st.error("Input data contains NaN or None values.")
            return None

        input_data = input_data.reshape(1, -1)
        prediction = model.predict(input_data)
        return int(prediction[0])

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Sidebar Navigation
st.sidebar.title("Navigation 🚀")
page = st.sidebar.radio("Navigate to:", ["Home 🏠", "Documentation 📚", "Predict 🔍", "Feedback 📝"])

# Home Page
if page == "Home 🏠":
    st.markdown("<div class='main-title'> Epilepsy Detection Model </div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Leverage AI to detect epileptic seizures from EEG signals.</div>", unsafe_allow_html=True)
    st.write("""
        - Navigate to the **Predict** page to input EEG signal features and receive predictions.
        - Visit the **Documentation** page for detailed usage instructions.
        - A model trained on state-of-the-art machine learning techniques powers this app.
    """)

# Documentation Page
elif page == "Documentation 📚":
    st.markdown("<div class='section-header'>📘 Documentation</div>", unsafe_allow_html=True)
    st.write("""
        - Input your features as a Python array on the **Predict** page.
        - Ensure that the array contains exactly 33 numerical features.
        - Example input: `[-108, -103, -96, -84, -79, -73, -61, -53, -43, -26, -9, 3, 9, 21, 20, 25, 30, 45, 47, 48, 56, 62, 64, 67, 67, 65, 59, 57, 65, 68, 78, 75, 59]`.
        - Click the **Make Prediction** button to analyze the features.
        - The result will show whether the signals are normal or epileptic.
    """)

# Prediction Page
elif page == "Predict 🔍":
    st.markdown("<div class='section-header'>🔍 Make a Prediction</div>", unsafe_allow_html=True)
    st.write("""
        Input the 33 EEG signal features in a Python array format below. Example:
        ```python
        [-108, -103, -96, -84, -79, -73, -61, -53, -43, -26, -9, 3, 9, 21, 20, 25, 30, 45, 47, 48, 56, 62, 64, 67, 67, 65, 59, 57, 65, 68, 78, 75, 59]
        ```
    """)

    # Input box for features
    input_features = st.text_input("Enter the 33 features (comma-separated, e.g., 1, 2, 3, ..., 33):")

    if st.button("Make Prediction"):
        try:
            features = eval(input_features)
            if len(features) != 33:
                st.error("Please enter exactly 33 features.")
            else:
                # Perform prediction
                prediction_binary = model.predict([features])[0]
                prediction_label = "Normal Brain Activity" if prediction_binary == 0 else "Epileptic Seizure Activity"

                # Prescription-style output
                st.markdown("""
                <div style="border: 2px solid #4CAF50; padding: 15px; border-radius: 10px; background-color: #f9f9f9;">
                    <h3 style="color: #2196F3; text-align: center;">🩺 Medical Report</h3>
                    <p style="font-size: 16px;"><b>Patient Information:</b></p>
                    <ul style="font-size: 16px;">
                        <li><b>EEG Signals Analyzed:</b> Provided by User</li>
                        <li><b>Number of Features:</b> 33 (validated)</li>
                    </ul>
                    <p style="font-size: 16px;"><b>Results:</b></p>
                    <ul style="font-size: 16px;">
                        <li><b>Condition:</b> <span style="color: #FF5722;"><b>{}</b></span></li>
                        <li><b>Recommendation:</b> {}</li>
                    </ul>
                    <p style="font-size: 16px; color: #555;">For further analysis or if symptoms persist, consult a neurologist.</p>
                </div>
                """.format(
                    prediction_label,
                    "Our Model predicted NORMAL brain ectivity. No immediate medical attention needed." if prediction_binary == 0 else "Our Model predicted ABNORMAL electrical activity. Seek professional medical advice promptly."
                ), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Invalid input. Please ensure the features are entered as a valid Python list. Error: {e}")

# Feedback Page
elif page == "Feedback 📝":
    st.markdown("<div class='section-header'>📝 User Feedback</div>", unsafe_allow_html=True)
    st.write("""
        We would love to hear your feedback to improve this app! Please provide your comments or suggestions below.
    """)

    # Text input for feedback
    feedback = st.text_area("Your feedback:")

    # Rating input (1-5 scale)
    rating = st.slider("Rate your experience (1-5)", min_value=1, max_value=5)

    # Submit button
    if st.button("Submit Feedback"):
        if feedback:
            st.success("Thank you for your feedback! 🙏")
            # You can store this feedback in a database or a file if you want to process it later
            with open("feedback.txt", "a") as f:
                f.write(f"Rating: {rating} | Feedback: {feedback}\n\n")
        else:
            st.error("Please provide your feedback before submitting.")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div class='footer'>Developed with ❤️ by Pakhi | Bioinformatics on the go 🚀</div>", unsafe_allow_html=True)
