import streamlit as st
import numpy as np

from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input


# ==========================================
# MODEL PATHS
# ==========================================

MODEL_PATH = "Brain_Tumor_VGG16_Model.h5"

MRI_VALIDATOR_PATH = "MRI_Validator_VGG16.h5"


IMAGE_SIZE = 128

MRI_THRESHOLD = 0.80


# ==========================================
# CLASS NAMES
# ==========================================

class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


# ==========================================
# STREAMLIT PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Brain Tumor MRI Detection",
    page_icon="🧠",
    layout="centered"
)


# ==========================================
# LOAD TUMOR MODEL
# ==========================================

@st.cache_resource
def load_brain_model():

    return load_model(
        MODEL_PATH,
        compile=False
    )


# ==========================================
# LOAD MRI VALIDATOR
# ==========================================

@st.cache_resource
def load_mri_validator():

    return load_model(
        MRI_VALIDATOR_PATH,
        compile=False
    )


model = load_brain_model()

mri_validator = load_mri_validator()


# ==========================================
# TITLE
# ==========================================

st.title(
    "🧠 Brain Tumor MRI Detection"
)

st.write(
    "Upload an MRI brain image to get the predicted result."
)


# ==========================================
# DISCLAIMER
# ==========================================

st.info(
    "⚠️ Educational/Research Demo: "
    "This model is not a medical diagnosis and should not "
    "be used as a substitute for a qualified medical professional."
)


# ==========================================
# FILE UPLOADER
# ==========================================

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)


# ==========================================
# PROCESS IMAGE
# ==========================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # --------------------------------------
    # DISPLAY UPLOADED IMAGE
    # --------------------------------------

    st.subheader(
        "Uploaded Image"
    )

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )


    # --------------------------------------
    # RESIZE IMAGE
    # --------------------------------------

    processed_image = image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )


    image_array = img_to_array(
        processed_image
    )


    # ======================================
    # STEP 1: MRI VALIDATION
    # ======================================

    validator_input = np.expand_dims(
        preprocess_input(
            image_array.astype(np.float32)
        ),
        axis=0
    )


    mri_probability = mri_validator.predict(
        validator_input,
        verbose=0
    )[0][0]


    mri_confidence = (
        mri_probability * 100
    )


    st.subheader(
        "Image Validation"
    )


    st.write(
        f"MRI Probability: "
        f"{mri_confidence:.2f}%"
    )


    # ======================================
    # STEP 2: REJECT NON-MRI
    # ======================================

    if mri_probability < MRI_THRESHOLD:

        st.error(
            "❌ Invalid Image"
        )

        st.warning(
            "Please upload MRI images only."
        )

        st.stop()


    # ======================================
    # STEP 3: MRI ACCEPTED
    # ======================================

    st.success(
        "✅ MRI Image Detected"
    )


    # ======================================
    # STEP 4: TUMOR MODEL INPUT
    # ======================================

    tumor_input = image_array / 255.0

    tumor_input = np.expand_dims(
        tumor_input,
        axis=0
    )


    # ======================================
    # STEP 5: TUMOR PREDICTION
    # ======================================

    predictions = model.predict(
        tumor_input,
        verbose=0
    )[0]


    predicted_index = np.argmax(
        predictions
    )


    predicted_class = class_names[
        predicted_index
    ]


    confidence = (
        predictions[predicted_index] * 100
    )


    # ======================================
    # STEP 6: DISPLAY RESULT
    # ======================================

    st.subheader(
        "Prediction"
    )


    if predicted_class == "notumor":

        result = "No Tumor"

    else:

        result = predicted_class.title()


    st.success(
        f"Prediction: {result}"
    )


    st.write(
        f"Tumor Model Confidence: "
        f"{confidence:.2f}%"
    )


    # ======================================
    # STEP 7: CLASS PROBABILITIES
    # ======================================

    st.subheader(
        "Class Probabilities"
    )


    for i, class_name in enumerate(
        class_names
    ):

        probability = (
            predictions[i] * 100
        )


        if class_name == "notumor":

            display_name = "No Tumor"

        else:

            display_name = class_name.title()


        st.write(
            f"{display_name}: "
            f"{probability:.2f}%"
        )


        st.progress(
            float(predictions[i])
        )