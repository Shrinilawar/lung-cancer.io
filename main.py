import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# Define class labels
class_labels = ['Adenocarcinoma', 'Large Cell Carcinoma', 'Normal', 'Squamous Cell Carcinoma']

# Load the trained model
model = tf.keras.models.load_model('trained_lung_cancer_model_final.h5')

# Function to preprocess the uploaded image
def load_and_preprocess_image(img_path, target_size=(350, 350)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Home section content
def home():
    st.markdown(
        """
        <div style="width: 100%; padding: 20px;">
            <h2 style="color:#333;">Lung Cancer Detection</h2>
            <p style="color:#555;">
                Lung cancer is one of the most common and serious types of cancer. 
                Early detection is crucial for effective treatment and better outcomes.
            </p>
        </div>
        """, unsafe_allow_html=True
    )
    
    # Add a Predict button that redirects to the Predict page using session state
    if st.button('Go to Predict Page'):
        st.session_state.page = "Predict"

# Predict section content
def predict():
    st.markdown(
        """
        <div style="width: 100%; padding: 20px;">
            <h2 style="color:#333;">Predict Lung Cancer Type</h2>
            <p style="color:#555;">Upload a lung CT scan image to classify it into one of the following categories:</p>
        </div>
        """, unsafe_allow_html=True
    )
    st.write(class_labels)

    # File uploader to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        # Load the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Save the uploaded image locally
        img_path = 'uploaded_image.png'
        image.save(img_path)

        # Preprocess the image
        preprocessed_image = load_and_preprocess_image(img_path)

        # Display the preprocessed image shape
        st.write(f"Preprocessed image shape: {preprocessed_image.shape}")

        # Make predictions
        predictions = model.predict(preprocessed_image)

        # Get the predicted class
        predicted_class = np.argmax(predictions[0])
        predicted_label = class_labels[predicted_class]

        # Display the prediction
        st.success(f"Prediction: **{predicted_label}**")

        # Display the prediction probabilities for each class
        st.write("Prediction probabilities:")
        for i, label in enumerate(class_labels):
            st.write(f"{label}: {predictions[0][i] * 100:.2f}%")

        # Plot the image with the predicted label as the title
        plt.imshow(image)
        plt.title(f"Predicted: {predicted_label}")
        plt.axis('off')
        st.pyplot(plt)

# About section content
def about():
    st.markdown(
        """
        <div style="width: 100%; padding: 20px;">
            <h2 style="color:#333;">About the Developer</h2>
            <p style="color:#555;">
                Developed by [Shrikumar,Prathamesh,and Team ]. 
                Passionate about AI and ML, I aim to leverage technology to improve healthcare outcomes.
            </p>
        </div>
        """, unsafe_allow_html=True
    )

# Main app function for Streamlit
def main():
    # Add a title and description
    st.markdown(
        """
        <style>
        body {
            background-image: 'Schedule a Consultation.png'; /* Add the actual URL of your background image */
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            padding: 20px;
        }
        h2 {
            font-size: 2rem; /* Responsive font size */
        }
        p {
            font-size: 1.2rem; /* Responsive font size */
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.title("Lung Cancer Detection App")
    st.markdown(
        """
        <h2 style="color:#FFA500;">Cancer is a part of our life, it's not our whole life.</h2>
        """, unsafe_allow_html=True
    )

    # Set up a session state for page navigation
    if 'page' not in st.session_state:
        st.session_state.page = "Home"

    # Sidebar for navigation
    with st.sidebar:
        page = st.selectbox("Select a page", ["Home", "Predict", "About"])
        st.session_state.page = page

    # Page content based on selection
    if st.session_state.page == "Home":
        home()
    elif st.session_state.page == "Predict":
        predict()
    elif st.session_state.page == "About":
        about()

    # Add footer
    st.markdown(
        """
        <style>
        footer {
            visibility: hidden;
        }
        .footer {
            visibility: visible;
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #262730;
            color: white;
            text-align: center;
            padding: 5px;
        }
        </style>
        <div class="footer">
            Developed by MedTech
        </div>
        """, unsafe_allow_html=True
    )

# Run the app
if __name__ == '__main__':
    main()