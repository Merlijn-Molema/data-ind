import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN

# ---------------------------------------------------------
# LOAD MODELS / PLACEHOLDER FUNCTIONS
# ---------------------------------------------------------

def load_autoencoder():
    # TODO: load your trained Keras/PyTorch autoencoder
    return None

AUTOENCODER = load_autoencoder()

def preprocess_image(img_pil, target_size=(64, 64)):
    img = img_pil.resize(target_size)
    img_arr = np.array(img) / 255.0
    img_arr = img_arr.astype("float32")
    return img_arr

def reconstruct(img_arr):
    # TODO: run through your autoencoder
    # return reconstructed_img_arr
    return img_arr  # placeholder

def compute_reconstruction_error(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)

def apply_gaussian_blur(img_arr, ksize=(15, 15)):
    return cv2.GaussianBlur(img_arr, ksize, 0)

def apply_pixelate_blur(img_arr, factor=8):
    h, w, c = img_arr.shape
    temp = cv2.resize(img_arr, (w//factor, h//factor), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    return pixelated

def detect_faces(img_arr):
    # Example face detector (Haar cascade)
    gray = cv2.cvtColor((img_arr * 255).astype("uint8"), cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

def selective_blur(img_arr, faces):
    out = img_arr.copy()
    for (x, y, w, h) in faces:
        roi = out[y:y+h, x:x+w]
        blurred = apply_pixelate_blur(roi, factor=12)
        out[y:y+h, x:x+w] = blurred
    return out

def add_noise(img_arr, noise_type="gaussian"):
    if noise_type == "gaussian":
        noise = np.random.normal(0, 0.1, img_arr.shape)
        return np.clip(img_arr + noise, 0, 1)
    elif noise_type == "salt_pepper":
        noisy = img_arr.copy()
        prob = 0.02
        rnd = np.random.rand(*img_arr.shape[:2])
        noisy[rnd < prob] = 0
        noisy[rnd > 1 - prob] = 1
        return noisy
    return img_arr

# ---------------------------------------------------------
# STREAMLIT CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Anomaly Detection Dashboard",
    layout="wide"
)

st.title("Anomaly Detection Dashboard – Autoencoder, Blurring & Noise Removal")
st.write("A complete demonstration of anomaly detection, image reconstruction, noise removal, and comparison models.")

tabs = st.tabs([
    "1. Introduction & Dataset",
    "2. Anomaly Detection",
    "3. Blurring",
    "4. Noise Removal",
    "5. Comparison Models",
    "6. Performance Metrics",
    "7. Examples & Failure Cases",
    "8. Technical Notes"
])

# ---------------------------------------------------------
# TAB 1 — INTRODUCTION & DATASET
# ---------------------------------------------------------
with tabs[0]:
    st.header("Introduction")
    st.write("""
    This dashboard demonstrates anomaly detection using an autoencoder trained on a unique custom dataset.
    You can explore model performance, anomaly classification, blurring techniques, noise removal, and comparison algorithms.
    """)
    
    st.subheader("Dataset Overview")
    st.write("""
    - Unique dataset collected for this project  
    - Image size: 64×64  
    - Preprocessing: resizing, normalization, splitting into train/val/test  
    - Includes both normal and anomalous samples  
    """)

    st.subheader("Workflow Diagram")
    st.image("workflow.png", caption="Overall workflow", use_column_width=True)

# ---------------------------------------------------------
# TAB 2 — ANOMALY DETECTION
# ---------------------------------------------------------
with tabs[1]:
    st.header("Autoencoder-Based Anomaly Detection")

    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")
        st.image(img_pil, caption="Uploaded Image", width=300)

        if st.button("Analyze Image"):
            img_arr = preprocess_image(img_pil)
            recon = reconstruct(img_arr)

            err = compute_reconstruction_error(img_arr, recon)

            threshold = 0.02  # TODO: use your real threshold

            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Original")
                st.image(img_arr, use_column_width=True)
            with col2:
                st.subheader("Reconstructed")
                st.image(recon, use_column_width=True)
            with col3:
                st.subheader("Difference (optional heatmap)")
                diff = np.abs(img_arr - recon)
                st.image(diff, use_column_width=True)

            st.subheader("Result")
            st.write(f"Reconstruction error: {err:.5f}")
            st.write(f"Threshold: {threshold:.5f}")
            
            if err > threshold:
                st.error("Anomaly Detected")
                st.session_state["last_img"] = img_arr
                st.session_state["is_anomaly"] = True
            else:
                st.success("Normal Image")
                st.session_state["is_anomaly"] = False

# ---------------------------------------------------------
# TAB 3 — BLURRING
# ---------------------------------------------------------
with tabs[2]:
    st.header("Blurring Techniques")

    if "last_img" not in st.session_state:
        st.warning("Run anomaly detection first.")
    else:
        img = st.session_state["last_img"]
        st.image(img, caption="Image for Blurring")

        if st.session_state.get("is_anomaly", False):
            st.info("Anomaly detected. You may apply blur.")

            blur_type = st.radio("Choose blur type", ["Gaussian", "Pixelate", "Selective"])

            if st.button("Apply Blur"):
                if blur_type == "Gaussian":
                    out = apply_gaussian_blur(img)
                elif blur_type == "Pixelate":
                    out = apply_pixelate_blur(img)
                else:
                    faces = detect_faces(img)
                    out = selective_blur(img, faces)
                
                st.image(out, caption="Blurred Output")
        else:
            st.info("This image is not an anomaly. Blurring disabled.")

# ---------------------------------------------------------
# TAB 4 — NOISE REMOVAL
# ---------------------------------------------------------
with tabs[3]:
    st.header("Noise Removal with Autoencoder")

    noise_choice = st.selectbox("Select noise type", ["gaussian", "salt_pepper"])
    img_noise = add_noise(np.zeros((64, 64, 3)), noise_type=noise_choice)  # replace with real sample

    if st.button("Generate Noisy Sample"):
        st.subheader("Noisy Image")
        st.image(img_noise)

        denoised = reconstruct(img_noise)

        st.subheader("Denoised Output")
        st.image(denoised)

        mse = compute_reconstruction_error(img_noise, denoised)
        st.write(f"MSE: {mse:.5f}")

# ---------------------------------------------------------
# TAB 5 — COMPARISON MODELS
# ---------------------------------------------------------
with tabs[4]:
    st.header("Comparison with Clustering Models")
    st.write("""
        Uses K-Means and DBSCAN for anomaly detection comparison based on latent vectors.
    """)

    st.subheader("Clustering Visualization (placeholder)")
    st.write("Add PCA/t-SNE plots here.")

# ---------------------------------------------------------
# TAB 6 — PERFORMANCE METRICS
# ---------------------------------------------------------
with tabs[5]:
    st.header("Performance Metrics")

    st.subheader("Training & Validation Loss")
    st.line_chart({"train_loss": [0.1,0.08,0.06], "val_loss": [0.11,0.09,0.07]})

    st.subheader("Error Distribution")
    st.write("Insert histogram comparing reconstruction errors of normal vs anomaly here.")

    st.subheader("Confusion Matrix")
    st.write("Insert confusion matrix plot here.")

# ---------------------------------------------------------
# TAB 7 — EXAMPLES & FAILURE CASES
# ---------------------------------------------------------
with tabs[6]:
    st.header("Examples and Failure Cases")

    st.subheader("Normal Examples")
    st.image(["normal1.png", "normal2.png"])

    st.subheader("Anomaly Examples")
    st.image(["anomaly1.png", "anomaly2.png"])

    st.subheader("Failure Cases")
    st.write("""
    - Example 1: Normal classified as anomaly due to high texture.
    - Example 2: Anomaly not detected due to similarity to training data.
    """)

# ---------------------------------------------------------
# TAB 8 — TECHNICAL NOTES
# ---------------------------------------------------------
with tabs[7]:
    st.header("Technical Explanation")

    st.subheader("Autoencoder Architecture")
    st.write("""
    - Input: 64×64×3  
    - Encoder: Conv → Conv → Dense  
    - Bottleneck: 64-dimensional latent vector  
    - Decoder: Dense → ConvTranspose → Output layer  
    """)

    st.subheader("Threshold Selection")
    st.write("Threshold chosen based on reconstruction error percentiles of validation set.")

    st.subheader("Color Feature Improvements")
    st.write("Model was trained using RGB channels and histogram features to improve anomaly separation.")
