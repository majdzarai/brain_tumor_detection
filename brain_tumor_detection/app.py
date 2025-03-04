import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import base64

# 🌙 Set Page Configuration
st.set_page_config(page_title="Brain Tumor Detection AI", page_icon="🧠", layout="wide")

# 🎯 Load the trained model
MODEL_PATH = "vgg16_best_model.h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# 🔹 Define class labels
class_labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# 🌍 Memory to store previous results
if "previous_predictions" not in st.session_state:
    st.session_state.previous_predictions = []

# 🎨 Custom CSS for Professional Styling
st.markdown("""
    <style>
        .stButton > button { background-color: transparent; color: white; font-size: 16px; border: none; padding: 8px 16px; transition: 0.3s; }
        .stButton > button:hover { color: #4CAF50; text-decoration: underline; }
        .stFileUploader { border: 2px dashed #4CAF50; padding: 10px; }
        .table-container { display: flex; flex-wrap: wrap; border: 1px solid #4CAF50; border-radius: 10px; padding: 10px; margin-bottom: 20px; }
        .column-left, .column-right { flex: 50%; padding: 10px; }
        .result-text { font-size: 18px; font-weight: bold; color: #4CAF50; text-align: center; }
        .description-box { background-color: #2C2C2C; padding: 12px; border-radius: 8px; text-align: left; margin-top: 10px; }
        .button-container { text-align: center; margin-top: 10px; }
        .icon-button { display: flex; flex-direction: column; align-items: center; justify-content: center; font-size: 14px; }
    </style>
    """, unsafe_allow_html=True)

# 🎭 Sidebar - Navigation Menu
st.sidebar.image("Glioma.gif", width=200)
st.sidebar.markdown("<p class='sidebar-title'>🔍 Explore the App</p>", unsafe_allow_html=True)
menu = st.sidebar.radio("📌 Select Page:", ["Home", "CNN Model Explanation", "VGG16 Model Explanation", "How It Works"])

# 📌 CNN Model Explanation
if menu == "CNN Model Explanation":
    st.title("🧠 Understanding CNN")
    st.write("""
    CNNs (Convolutional Neural Networks) are deep learning models specifically designed for image recognition tasks. They consist of:
    - Convolutional layers to extract image features.
    - Pooling layers to reduce spatial dimensions.
    - Fully connected layers to classify the image.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/63/Typical_cnn.png", caption="CNN Architecture", use_column_width=True)

# 📌 VGG16 Model Explanation
elif menu == "VGG16 Model Explanation":
    st.title("📖 Understanding VGG16")
    st.write("""
    VGG16 is a deep CNN model with **16 layers**, pre-trained on millions of images. Features:
    - Uses small **3x3 convolutions** to learn complex patterns.
    - Well-suited for medical image analysis.
    """)
    st.image("https://www.researchgate.net/publication/346259768/figure/fig8/AS:961803395801094@1606323207263/GG-16-CNN-model-architecture-layer-wise.jpg", caption="VGG16 Architecture", use_column_width=True)

# 📌 How It Works
elif menu == "How It Works":
    st.title("🎯 How The Brain Tumor AI Works")
    st.write("""
    1️⃣ **Upload an MRI Scan**.
    2️⃣ **Preprocessing**: Resize, normalize, and convert image to an array.
    3️⃣ **Prediction**: The trained VGG16 model classifies the image.
    4️⃣ **Confidence Score** is displayed.
    """)
    st.image("model.png", caption="AI Model Workflow", use_column_width=True)

# 📌 View Previous Predictions
if menu == "View Previous Predictions":
    st.title("💾 Previous Predictions")
    if not st.session_state.previous_predictions:
        st.info("No previous predictions yet.")
    else:
        for prev in st.session_state.previous_predictions:
            st.markdown(f"**{prev['tumor_type']}** - {prev['filename']}")
            st.image(prev["image"], width=200)
    if st.button("🗑️ Clear Memory"):
        st.session_state.previous_predictions.clear()
        st.experimental_rerun()

# 📌 Home Page - Main Model UI
if menu == "Home":
    st.title("🧠 Brain Tumor Detection AI")
    st.write("Upload an **MRI scan** and the model will predict the tumor type.")

    # 🚀 **Buttons Section with Labels**
    colA, colB, colC, colD = st.columns([1, 1, 1, 1])

    with colA:
        if st.button("🔄 Refresh", help="Refresh Page", key="refresh"):
            st.rerun()

    with colB:
        if st.button("📷 Predict One", help="Predict One Image", key="predict_one"):
            predict_one = True

    with colC:
        if st.button("📂 Predict Multiple", help="Predict Multiple Images", key="predict_multiple"):
            predict_multiple = True

    with colD:
        if st.button("💾 View Memory", help="View Previous Predictions", key="memory"):
            st.session_state.menu = "View Previous Predictions"
            st.experimental_rerun()

    # 📤 Upload MRI Scan
    uploaded_files = st.file_uploader("📤 Upload MRI Image(s)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        st.write("### 🏥 Predictions:")
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            image = image.convert("RGB")

            # Convert image to base64 for proper display in HTML
            buffered = Image.new("RGB", image.size, (0, 0, 0))
            buffered.paste(image)
            img_data = buffered.resize((200, 200))
            img_buffer = img_data.convert("RGB")
            with open("temp_img.png", "wb") as temp_file:
                img_buffer.save(temp_file, format="PNG")
            with open("temp_img.png", "rb") as temp_file:
                img_base64 = base64.b64encode(temp_file.read()).decode()

            # 🛠 Preprocess Image
            img = image.resize((224, 224))
            img = img_to_array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            # 🔍 Make Prediction
            predictions = model.predict(img)
            predicted_class = np.argmax(predictions)
            tumor_type = class_labels[predicted_class]

            # Store in session for memory
            st.session_state.previous_predictions.append({
                "filename": uploaded_file.name,
                "tumor_type": tumor_type,
                "image": image
            })

            # 📌 **Table-Like Layout**
            st.markdown(f"""
            <div class='table-container'>
                <div class='column-left'>
                    <img src="data:image/png;base64,{img_base64}" width="200" style="border-radius: 10px;">
                </div>
                <div class='column-right'>
                    <p class='result-text'>🧠 {tumor_type}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

                        # 📌 **Tumor Description Box**
            if predicted_class == 0:  # Glioma
                description = """
                Gliomas are brain tumors that originate from **glial cells**, which support and protect neurons. 
                They can be **malignant or benign** and often affect the brain's function over time. 
                Common symptoms include **headaches, seizures, and memory loss**.
                """
            elif predicted_class == 1:  # Meningioma
                description = """
                Meningiomas are **tumors that develop in the membranes surrounding the brain and spinal cord**. 
                They are usually **benign**, but their growth can cause **pressure on the brain**, leading to symptoms like **vision problems, headaches, and seizures**.
                """
            elif predicted_class == 2:  # No Tumor
                description = """
                **Great news! No brain tumor was detected.**  
                Your MRI scan shows no abnormalities, and everything looks fine. 
                However, maintaining a **healthy lifestyle** and having **regular check-ups** is always recommended!
                """
            elif predicted_class == 3:  # Pituitary
                description = """
                Pituitary tumors form in the **pituitary gland**, a key part of the body that controls **hormone production**. 
                These tumors can lead to **hormonal imbalances**, affecting **growth, metabolism, and vision**.
                """

            # 📌 **Show Description Below Prediction**
            st.markdown(f"""
            <div class='description-box' style="border-left: 5px solid #4CAF50; background-color: #f9f9f9; padding: 15px; border-radius: 8px; margin-top: 10px;">
                <h4 style="color: #333;">📋 About {tumor_type}</h4>
                <p style="color: #555;">{description}</p>
            </div>
            """, unsafe_allow_html=True)


            # 📌 **"How to Deal With It?" Button**
            if st.button(f"🩺 How to Deal With {tumor_type}? {uploaded_file.name}", key=f"deal_{uploaded_file.name}"):
                if predicted_class == 0:  # Glioma
                    st.warning("⚠️ **Glioma Tumor Advice**")
                    st.write("""
                    - Consult a **neurosurgeon** immediately.
                    - MRI scans should be done **regularly** to monitor growth.
                    - Treatment options include **surgery, radiation, and chemotherapy**.
                    - Maintain a **healthy diet and active lifestyle**.
                    """)
                elif predicted_class == 1:  # Meningioma
                    st.warning("🧠 **Meningioma Tumor Advice**")
                    st.write("""
                    - Most meningiomas are **benign**, but medical consultation is required.
                    - If symptoms appear, **surgery or radiation therapy** may be recommended.
                    - Regular checkups help track growth.
                    """)
                elif predicted_class == 2:  # No Tumor
                    st.success("✅ **You are OK! No Tumor Detected!**")
                    st.write("No medical intervention needed. Stay healthy!")
                elif predicted_class == 3:  # Pituitary
                    st.error("⚠️ **Pituitary Tumor Advice**")
                    st.write("Consult an **endocrinologist** for hormone monitoring.")

# 🚀 Footer
st.markdown("---")
st.markdown("<p style='text-align:center;'>Developed by <strong>Majd Zarai</strong> | Powered by TensorFlow & Streamlit 🚀</p>", unsafe_allow_html=True)
