import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


MODEL_PATH = "Brain_Tumor_VGG16_Model.h5"
IMAGE_SIZE = 128

class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


st.set_page_config(
    page_title="Brain Tumor MRI Detection",
    page_icon="🧠",
    layout="centered"
)


@st.cache_resource
def load_brain_model():
    return load_model(
        MODEL_PATH,
        compile=False
    )


model = load_brain_model()


st.title("🧠 Brain Tumor MRI Detection")
st.write(
    "Upload an MRI brain image to get the predicted result."

)

st.info(
    "⚠️ Educational/Research Demo: "
    "This model is not a medical diagnosis and should not "
    "be used as a substitute for a qualified medical professional."
)


uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded MRI")
    st.image(
        image,
        caption="MRI Image",
        use_container_width=True
    )


    processed_image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    image_array = img_to_array(processed_image)

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    predictions = model.predict(
        image_array,
        verbose=0
    )[0]


    predicted_index = np.argmax(predictions)

    predicted_class = class_names[predicted_index]

    confidence = predictions[predicted_index] * 100


    st.subheader("Prediction")


    if predicted_class == "notumor":
        result = "No Tumor"
    else:
        result = predicted_class.title()


    st.success(
        f"Prediction: {result}"
    )

    st.write(
        f"Confidence: {confidence:.2f}%"
    )


    st.subheader("Class Probabilities")


    for i, class_name in enumerate(class_names):

        probability = predictions[i] * 100

        if class_name == "notumor":
            display_name = "No Tumor"
        else:
            display_name = class_name.title()

        st.write(
            f"{display_name}: {probability:.2f}%"
        )

        st.progress(
            float(predictions[i])
        )