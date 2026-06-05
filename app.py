# app.py
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ------------------------------
# Load Keras model from HuggingFace
# ------------------------------

@st.cache_resource
def load_keras_model(model_version = "HelmNet_Image_Classification.v0.h5"):
    model_path = hf_hub_download(
        repo_id="DanielLevenstein/HelmNet_Image_Classification",
        filename=model_version,
    )
    model = load_model(model_path)
    return model

# ------------------------------
# Streamlit UI
# ------------------------------

st.title("Helmet Detection")
st.markdown("""## Data Quality Case Study
In this project I compare the performance of 4 CNN helmet detection models and provide an interactive streamlit app allowing you to evaluate your own images against these 4 models. 
### Training Dataset

- [Dataset0](https://github.com/DanielLevenstein/HelmNet_ImageProcessing_UT_ML_Project6/)       : Great Learning Private Classification Dataset Size ~600         
- [Dataset1](https://www.kaggle.com/datasets/rajeevsekar21/on-vehicle-helmet-detection-dataset) : Kaggle Classification Dataset Size ~500       
- [Dataset2](https://www.kaggle.com/datasets/andrewmvd/helmet-detection)                        : Kaggle Bounding Boxes Dataset Size ~250        
- [Dataset3](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)                      : Kaggle Bounding Boxes Dataset Size ~3000  

""")

st.markdown("""
### Model Validation:
Model Validation has been moved to it's own standalone notebook which uses a holdout dataset from Dataset2, and Dataset3, and can be found on GitHub along with the training notebooks for Dataset 1 to 3. 

* All four models are evaluated based on from a hold-out dataset from Dataset2, and Dataset3
* Because Dataset2 is significantly smaller than Dataset3, a sample of 50 images was taken from each holdout set for evaluation.
* When evaluating Model0 and Model1, the validation data is resized to 200x200 prior to evaluation.

(Previously this app validated images based on a dataset of images downloaded from the web.)

#### Summary Of Cross Validation Testing:
    
[Full Validation Notebook](https://github.com/DanielLevenstein/Helmet_CNN_Model_Training_Notebooks/blob/main/model_cross_validation.ipynb)

| Model  | Accuracy | Precision |   Recall | F1-Score |
| :----- | -------: | --------: | -------: | -------: |
| Model3 |     0.93 |  0.955056 | 0.965909 | 0.960452 |
| Model2 |     0.93 |  0.926316 | 1.000000 | 0.961749 |
| Model1 |     0.70 |  0.983333 | 0.670455 | 0.797297 |
| Model0 |     0.76 |  0.890244 | 0.829545 | 0.858824 |
""")
st.markdown("### Helmet Detection Demo:")
model0 = load_keras_model("HelmNet_Image_Classification.v0.h5")
model1 = load_keras_model("HelmNet_Image_Classification.v1.h5")
model2 = load_keras_model("HelmNet_Image_Classification.v2.h5")
model3 = load_keras_model("HelmNet_Image_Classification.v3.h5")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


def get_prediction_class(confidence, helmet_threshold, no_helmet_threshold):
    if confidence >= helmet_threshold:
        predicted_class = "Helmet"
    elif confidence <= no_helmet_threshold:
        predicted_class = "No Helmet"
    else:
        predicted_class = "Unknown"
    return predicted_class

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', width=200)

    # Preprocess image
    # Adjusts size to your model's expected input shape
    img_resized = img.resize((200, 200))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize if required

    # Predict
    helmet_threshold = 0.50
    no_helmet_threshold = 0.50
    # Define your class names (adjust if needed)
    prediction = model0.predict(img_array)

    confidence = np.max(prediction)
    predicted_class_index = np.argmax(prediction)

    if confidence >= helmet_threshold:
        predicted_class = "Helmet"
    elif confidence <= no_helmet_threshold:
        predicted_class = "No Helmet"
    else:
        predicted_class = "Unknown"

    if predicted_class == "No Helmet":
        confidence = 1 - confidence

    st.markdown("## Analysis of uploaded image")
    prediction1 = model1.predict(img_array)
    confidence1 = np.max(prediction1)
    predicted_class1 = get_prediction_class(confidence1, helmet_threshold, no_helmet_threshold)

    if(predicted_class1 == "No Helmet"):
        confidence1 = 1 - confidence1


    # Adding model2 prediction
    img_resized = img.resize((100, 100))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize if required

    prediction2 = model2.predict(img_array)
    confidence2 = np.max(prediction2)
    predicted_class2 = get_prediction_class(confidence2, helmet_threshold, no_helmet_threshold)


    prediction3 = model3.predict(img_array)
    confidence3 = np.max(prediction3)
    predicted_class3 = get_prediction_class(confidence3, helmet_threshold, no_helmet_threshold)

    st.markdown(f"""
    - Model0 Prediction: **{predicted_class}** Confidence: **{confidence:.2f}**
    - Model1 Prediction: **{predicted_class1}**  Confidence: **{confidence1:.2f}**
    - Model2 Prediction: **{predicted_class2}** Confidence: **{confidence2:.2f}**
    - Model3 Prediction: **{predicted_class3}**  Confidence: **{confidence3:.2f}**
    """)

    st.write("The performance of Model2 and Model3 were close. Model3 had the highest precision, whereas the F1 scores and accuracy scores were tied between the Model2, and Model3.")

st.markdown("""

#### Model0 Dataset Summary: 
- Original Training Notebook: [HelmNet_ImageProcessing_Notebook](https://github.com/DanielLevenstein/HelmNet_ImageProcessing_UT_ML_Project6/)
- Images in this dataset were resized to 200x200 prior to training.
- The final accuracy of this model was ~98% when measured against model0 validation data.
- When the model was run against unseen data, it classified many images as containing helmets which did not have them.
- I set a confidence threshold of 95% in an attempt to counteract false positives.
- This appeared to improve clear performance in the notebook, but it did not eliminate false positives on unseen data.
- The confidence threshold has been set back to .50 in the latest version of the streamlit app.

[Dataset 0 Training Notebook: HelmNet_ImageProcessing_UT_ML_Project6](https://github.com/DanielLevenstein/HelmNet_ImageProcessing_UT_ML_Project6/)
""")
st.image('charts/dataset0_sample_images.png')
st.markdown("""
#### Model1 Dataset Summary:

- Two of the misclassified images in the dataset appeared to be duplicates, which makes me worry the dataset has other duplicate images.
- This dataset has an almost equal balance between no helmet samples and helmet samples.
- The images in this dataset only contain two individual riders, which is going to harm the generalizability of the training data.
- Images in this dataset are taken by hand and then resized to 200x200 prior to training.
- Many of the training images were upside down or sideways in this dataset.

[Dataset 1 Training Data: on-vehicle-helmet-detection-dataset](https://www.kaggle.com/datasets/rajeevsekar21/on-vehicle-helmet-detection-dataset)
""")
st.image('charts/dataset1_sample_images.png')
st.markdown("""
#### Model2 Dataset Summary:
- Model2 has a high Recall score but a lower precision and accuracy score.
- Several of these falsely categorized images contain non-helmet headcoverings which could explain these results.
- This dataset contains ~250 images and has more samples with helmets than without.
- This training dataset was generated from bounding box data in a vehicle helmet dataset.
- All images are cropped at exactly 100x100 pixels, and samples with features smaller than 50 px are removed from the dataset.

[Dataset 2 Training Data: helmet-detection](https://www.kaggle.com/datasets/andrewmvd/helmet-detection)
""")
st.image('charts/dataset2_sample_images.png')

st.markdown("""
#### Model3 Dataset Summary:
- Model3 has the highest overall accuracy so far and does not appear to display the precision issues we saw for Model2.
- Some of the mis-categorized images in this dataset contain helmet images in the background, which could explain why they got miscategorized.
- This dataset contains ~3000 images and has more samples with helmets than without.
- This training dataset was generated from bounding box data construction site photographs.
- All images are cropped at exactly 100x100 pixels, and samples with features smaller than 50 px are removed from the dataset.

[Dataset 3 Training Data: hard-hat-detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)
""")
st.image('charts/dataset3_sample_images.png')
