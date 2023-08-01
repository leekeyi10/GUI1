import streamlit as st
import pandas as pd
from PIL import Image
import os
from session_state import SessionState

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score
import tensorflow as tf
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input, concatenate, Dense, Dropout
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from keras.applications.resnet50 import preprocess_input as preprocess_resnet_input
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input, concatenate
from keras.layers import Dense, Dropout
from keras.layers import Activation, Dense
from skimage.transform import rotate
from skimage.util import random_noise
from skimage.transform import resize, rotate
from skimage.util import random_noise
from skimage.filters import unsharp_mask
import imgaug.augmenters as iaa
import time
import processing

# Read the CSV file
data = pd.read_csv('Final Dataset/Test_Train_classes.csv')
folder_path = 'Final Dataset'  # Replace with the path to your folder

# Create an instance of SessionState for session state persistence
session_state = SessionState(accuracy=None, precision=None, recall=None, confusion_matrix=None)

def extract_logo_context_features(image):
    if len(image.shape) == 2:  # Grayscale image
        # Convert grayscale image to 8-bit depth
        image = cv2.convertScaleAbs(image)

        # Example implementation for grayscale image processing
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    elif len(image.shape) == 3 and image.shape[2] == 3:  # RGB image
        # Convert RGB image to 8-bit depth
        image = cv2.convertScaleAbs(image)

        # Example implementation for RGB image processing
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    else:
        raise ValueError("Unsupported image format. Only grayscale and RGB images are supported.")

def extract_image_pixel_features(image):
    # Replace with your implementation for image pixel features
    # Example implementation:
    # Flatten and normalize the pixel intensities as a feature
    flattened_image = image.flatten()
    normalized_image = flattened_image / 255.0
    return normalized_image

def enhance_image(image):
    # Apply sharpening filter to enhance image clarity
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

def pixelize(image, factor=0.3):
    height, width, _ = image.shape
    new_height = int(height * factor)
    new_width = int(width * factor)
    return resize(resize(image, (new_height, new_width), anti_aliasing=True), (height, width), anti_aliasing=True)
def calculate_accuracy(predicted_label, ground_truth_label):
    return predicted_label == ground_truth_label

# Create the Streamlit app
def main():
    tab1, tab2, tab3, tab4 = st.tabs(["Data", "Model", "Robustness","Logo Detection"])

    with tab1:
        st.header("Raw Data")
        st.write(data)

        st.header("Image Example (Top 6)")
        image_files = os.listdir(folder_path)
        image_files = image_files[:6]  # Get the first 6 image files

        images_per_row = 2
        num_images = len(image_files)
        num_rows = (num_images - 1) // images_per_row + 1

        for i in range(num_rows):
            row_images = image_files[i * images_per_row: (i + 1) * images_per_row]
            row = st.columns(images_per_row)

            for j, filename in enumerate(row_images):
                image_path = os.path.join(folder_path, filename)
                image = Image.open(image_path)
                with row[j]:
                    st.image(image, caption=filename)

    with tab2:
        st.header("ResNet Model")


        from sklearn.utils import column_or_1d
        df_2 = pd.read_csv('Test_Train_classes.csv')

        images = []
        features = []
        for filename in df_2['Filename']:
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (299, 299))
            img = preprocess_resnet_input(img)
            images.append(img)

            context_features = extract_logo_context_features(img)
            pixel_features = extract_image_pixel_features(img)
            features.append(np.concatenate([context_features, pixel_features]))

        images = np.array(images)
        features = np.array(features)

        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(df_2['Label'])

        # Convert labels to a 1-dimensional array using ravel()
        labels = np.ravel(labels)
        labels = column_or_1d(labels, warn=True)

        X_train_images, X_test_images, X_train_features, X_test_features, y_train, y_test = train_test_split(
            images, features, labels, test_size=0.2, random_state=42)
        y_train = y_train.ravel()
        y_test = y_test.ravel()

        # Adjust the number of features based on your actual data
        num_features = 268715  # Replace with the actual number of features you have

        # Define the image model (using InceptionNet)
        image_input = Input(shape=(299, 299, 3))
        base_model = ResNet50(weights='imagenet', include_top=False)
        image_features = base_model(image_input)
        image_features = GlobalAveragePooling2D()(image_features)

        # Define the features model
        features_input = Input(shape=(num_features,))
        features_features = Dense(128, activation='relu')(features_input)

        # Combine the image and features models
        combined_features = concatenate([image_features, features_features])
        combined_features = Dense(128, activation='relu')(combined_features)
        combined_features = Dropout(0.5)(combined_features)
        predictions = Dense(1, activation='sigmoid')(combined_features)

            # Create the final model
            # model = Model(inputs=[image_input, features_input], outputs=predictions)
            #
            # model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
            # model.fit([X_train_images, X_train_features], y_train, batch_size=32, epochs=10, validation_split=0.2)
            #
            # model.save('my_model_ResNet50.h5')

        loaded_model = tf.keras.models.load_model('my_model_ResNet50.h5')

        # Record the start time
        start_time = time.time()
        y_pred = loaded_model.predict([X_test_images, X_test_features])
        # Record the end time
        end_time = time.time()
        # Calculate the runtime
        runtime = end_time - start_time

        y_pred = np.round(y_pred).astype(int)
        accuracy = accuracy_score(y_test, y_pred)

        class_labels = ['Fake', 'Genuine']
        precision = precision_score(y_test, y_pred, labels=class_labels)
        recall = recall_score(y_test, y_pred, labels=class_labels)

        confusionmatrix = confusion_matrix(y_test, y_pred)

        st.session_state['accuracy'] = accuracy
        st.session_state['precision'] = precision
        st.session_state['recall'] = recall
        st.session_state['confusion_matrix'] = confusionmatrix

        st.write("Model inference runtime: ", round(runtime, 4), " seconds")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("Accuracy: ", st.session_state['accuracy'].round(2))

        with col2:
            st.write("Precision: ", st.session_state['precision'].round(2))

        with col3:
            st.write("Recall: ", st.session_state['recall'].round(2))

        cm = st.session_state['confusion_matrix']
        # Plot the confusion matrix using Matplotlib and Seaborn
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')

        # Display the plot using st.pyplot()
        st.pyplot(fig)

    with tab3:
        st.header("Robustness")
        logo_folder_path = 'Valid'
        logo_files = [file for file in os.listdir(logo_folder_path) if file.endswith(('.jpg', '.png', '.jpeg'))]
        selected_logo_file = st.selectbox("Select an image", logo_files)
        image_path = os.path.join(logo_folder_path, selected_logo_file)

        def apply_transformations_and_predict(loaded_model, label_encoder, image_path, preprocess_input, enhance_image,
                                              extract_logo_context_features, extract_image_pixel_features):
            test_image = cv2.imread(image_path)
            test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
            test_image = cv2.resize(test_image, (299, 299))
            # test_image = preprocess_input(test_image)
            test_image_normalized = test_image / 255.0
            # Apply transformations and make predictions
            transformations = [
                ("Original", test_image),
                ("Rotated", rotate(test_image_normalized, angle=30, mode='constant', cval=1.0)),
                ("Noisy", random_noise(test_image_normalized, var=0.1 ** 2)),
                ("Flipped Horizontally", np.fliplr(test_image)),
                ("Flipped Vertically", np.flipud(test_image)),
                ("Darkened", np.clip(test_image_normalized ** 1.5, 0.0, 1.0)),
                ("Brightened", np.clip(test_image_normalized ** 0.6, 0.0, 1.0)),
                ("Sharpened", unsharp_mask(test_image, radius=1, amount=1)),
                ("Blurred", iaa.GaussianBlur(sigma=1.0).augment_image(test_image.astype(np.uint8))),
                ("Changed Aspect Ratio",
                 iaa.Affine(scale={"x": 1.0, "y": 3.0}).augment_image(test_image.astype(np.uint8))),
                ("Opacity", iaa.BlendAlpha(0.7, iaa.Add(100)).augment_image(test_image.astype(np.uint8))),
                ("Pixelized", pixelize(test_image, factor=0.3)),
                ("Scaled", iaa.Affine(scale=0.5).augment_image(test_image.astype(np.uint8)))
            ]
            for transform_name, transformed_image in transformations:
                # Preprocess the transformed image
                transformed_image = enhance_image(transformed_image)
                transformed_features = extract_logo_context_features(transformed_image)
                transformed_pixel_features = extract_image_pixel_features(transformed_image)

                # Check if either feature extraction resulted in an error
                if transformed_features is None or transformed_pixel_features is None:
                    st.write(f"Skipping transformation: {transform_name}")
                    continue

                # Concatenate the features
                transformed_features = np.concatenate([transformed_features, transformed_pixel_features])
                transformed_features = np.expand_dims(transformed_features, axis=0)

                # Preprocess and normalize the transformed image for display
                # transformed_image_display = transformed_image.astype(float) * 255.0
                # transformed_image_display = np.clip(transformed_image_display, 0.0, 255.0).astype(np.uint8)

                # Make predictions on the transformed image
                prediction = loaded_model.predict(
                    [np.expand_dims(transformed_image, axis=0), transformed_features])
                prediction_label = label_encoder.inverse_transform(np.round(prediction).astype(int))[0]

                # Display the image and its prediction using Streamlit
                st.image(transformed_image, use_column_width=True, clamp=True)
                st.write(f"<center>Transform: {transform_name}</center>", unsafe_allow_html=True)
                st.write(f"<center>Prediction: {prediction_label}</center>", unsafe_allow_html=True)
                st.markdown("<hr>", unsafe_allow_html=True)

        apply_transformations_and_predict(loaded_model, label_encoder, image_path, preprocess_input,
                                          enhance_image, extract_logo_context_features,
                                          extract_image_pixel_features)
    with tab4:
        st.title("Logo Detection")

        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        continue_button_placeholder = st.empty()
        prediction_placeholder = st.empty()

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            if continue_button_placeholder.button("Click Here To View Prediction"):
                st.session_state['uploaded_file'] = uploaded_file
                st.session_state['page'] = 'prediction'
                continue_button_placeholder.empty()

                prediction_placeholder.title("Prediction")

                if 'page' in st.session_state and st.session_state['page'] == 'prediction':
                    st.title("Prediction")
                    uploaded_image = st.session_state.get('uploaded_file')
                    if uploaded_image:
                        processing.process_image(uploaded_image, prediction_placeholder)
                        if uploaded_image:
                            # Read the uploaded image as a BytesIO object
                            pil_image = Image.open(uploaded_image)

                            # Convert the PIL image to a numpy array
                            test_image_1 = np.array(pil_image)

                            # Perform image processing on the numpy array
                            test_image_1 = cv2.cvtColor(test_image_1, cv2.COLOR_RGB2BGR)
                            test_image_1 = cv2.resize(test_image_1, (299, 299))
                            test_image_1 = preprocess_input(test_image_1)
                            test_image_1 = enhance_image(test_image_1)
                            test_features_1 = extract_logo_context_features(test_image_1)
                            test_pixel_features_1 = extract_image_pixel_features(test_image_1)

                            test_features_1 = np.concatenate([test_features_1, test_pixel_features_1])
                            test_features_1 = np.expand_dims(test_features_1, axis=0)

                            prediction_1 = loaded_model.predict([np.expand_dims(test_image_1, axis=0), test_features_1])
                            prediction_label_1 = label_encoder.inverse_transform(np.round(prediction_1).astype(int))[0]
                            st.write("Prediction: ",  prediction_label_1)

                            # Load the CSV file containing image paths and their corresponding labels
                            # The relative path to the CSV file is 'data/labels.csv'
                            df_labels = pd.read_csv('Valid/_classes.csv')

                            # Get the ground truth label for the uploaded image based on its file path
                            # Replace 'uploaded_image_file_path.jpg' with the actual file path of the uploaded image
                            ground_truth_label = \
                            df_labels[df_labels['filename'] == uploaded_image.name]['Label'].values[0]

                            accuracy = calculate_accuracy(prediction_label_1, ground_truth_label)
                            st.write("Accuracy: {:.2%}".format(accuracy))


if __name__ == '__main__':
    main()
