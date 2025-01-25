import streamlit as st
import requests
import torch
from PIL import Image
import cv2
import numpy as np
import os
from skimage.color import lab2rgb
from advanced_model import MainModel, load_pretrained_generator, load_main_model_for_inference  # Import from advanced_model.py
from model import MainModel as BasicMainModel  # Import basic model from model.py
from advanced_model import build_res_unet

# GitHub release URL
GITHUB_API_URL = "https://github.com/augustowski/ImageColorization_ADL_W24/releases/download/trained/"

# Map model names to GitHub release files
model_mapping = {
    "Basic Colorization Model": "model_run_4_state_dict.pth",
    "Advanced Colorization Model": "model_run_5_state_dict.pth",
}

# Set the local model directory
MODEL_DIR = "models2"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load model based on selection
def load_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(MODEL_DIR, model_mapping[model_name])

    # Check if model exists locally, download if necessary
    if not os.path.exists(model_path):
        st.write(f"Downloading {model_name} from GitHub...")
        model_url = f"{GITHUB_API_URL}{model_mapping[model_name]}"
        response = requests.get(model_url, stream=True)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                f.write(response.content)
            st.success(f"{model_name} downloaded successfully!")
        else:
            st.error(f"Failed to download {model_name}. Status code: {response.status_code}")
            st.stop()

    # Conditional loading for Basic vs Advanced model
    if model_name == "Basic Colorization Model":
        model = BasicMainModel()  # Initialize the basic model
        model.load_state_dict(torch.load(model_path, map_location=device))  # Load weights
        model.to(device)
    elif model_name == "Advanced Colorization Model":
        model = load_main_model_for_inference(model_path, device)  # Load advanced model (only generator)
    else:
        raise ValueError(f"Unknown model selection: {model_name}")

    model.eval()
    return model, device

# Preprocess input image
def preprocess_image(image):
    grayscale_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    input_image = cv2.resize(grayscale_image, (256, 256))
    input_image = input_image / 255.0
    input_tensor = (
        torch.tensor(input_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    )
    return input_tensor

# Postprocess model output
def postprocess_output(output_tensor, grayscale_tensor, original_size):
    """
    Converts the model's output tensor (AB channels) and grayscale input tensor (L channel) 
    into an RGB image and resizes it to the original image dimensions.
    Args:
        output_tensor (torch.Tensor): The model's output tensor with shape (1, 2, H, W).
        grayscale_tensor (torch.Tensor): The grayscale input tensor with shape (1, 1, H, W).
        original_size (tuple): Original image dimensions (width, height).
    Returns:
        np.ndarray: RGB image in the range [0, 255] and dtype uint8.
    """
    # Squeeze the tensors and move to CPU
    ab = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Shape: (H, W, 2)
    l = grayscale_tensor.squeeze(0).squeeze(0).cpu().numpy()      # Shape: (H, W)

    # Combine L and AB channels
    lab_image = np.zeros((l.shape[0], l.shape[1], 3), dtype=np.float32)
    lab_image[..., 0] = l * 100  # Scale L channel to range [0, 100]
    lab_image[..., 1:] = ab * 128  # Scale AB channels to range [-128, 127]

    # Convert LAB to RGB
    rgb_image = lab2rgb(lab_image)  # Output range [0, 1]
    rgb_image = (rgb_image * 255).astype(np.uint8)  # Convert to range [0, 255]

    # Resize the colorized image to match the original size
    rgb_image_resized = Image.fromarray(rgb_image).resize(original_size, Image.BICUBIC)
    return np.array(rgb_image_resized)


# Streamlit app configuration
st.title("Image Colorization App")
st.sidebar.title("Settings")

# Model selection
st.sidebar.subheader("Select a Colorization Model")
selected_model = st.sidebar.selectbox(
    "Model",
    options=["", "Basic Colorization Model", "Advanced Colorization Model"],
)

# Image upload
st.sidebar.subheader("Upload a Grayscale Image")
uploaded_file = st.sidebar.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if selected_model and uploaded_file:
    st.sidebar.success(f"Model: {selected_model}")
    st.sidebar.success(f"File: {uploaded_file.name}")

    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Grayscale Image", use_container_width=True)

    clicked = st.button("Colorize Image")

    if clicked:
        # Load the selected model
        model, device = load_model(selected_model)
    
        # Preprocess the uploaded image
        image = Image.open(uploaded_file)
        original_size = image.size  # Store the original dimensions (width, height)
        input_tensor = preprocess_image(image).to(device)
    
        # Perform inference
        with torch.no_grad():
            output_tensor = model(input_tensor)
    
        # Postprocess and display the output
        colorized_image = postprocess_output(output_tensor, input_tensor, original_size)
        st.image(colorized_image, caption="Colorized Image", use_container_width=True)

else:
    st.write("Please select a model and upload an image to start colorizing.")