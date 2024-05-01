import streamlit as st
from PIL import Image

# import torch
import torchvision.transforms as transforms
from aws_controls import AwsControl
from config import config

# from model import ResNetModel  # Assuming you have defined your model in a separate file named model.py

# Load the model
# model = ResNetModel()
# Load the weights
# model.load_state_dict(torch.load("path_to_your_trained_model_weights.pth"))
# model.eval()

# Define transformation for the input image
transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ]
)


# Define a function to make predictions
def predict(image):
    # Apply transformations
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    # Perform prediction
    # with torch.no_grad():
    # outputs = model(image_tensor)
    # _, predicted = torch.max(outputs, 1)
    # return predicted.item()
    return image_tensor


# Streamlit app
def main():
    st.title("Image Classification")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        image.save("uploaded_img.jpg")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Classify"):
            prediction = predict(image)
            st.write(f"Prediction: Class {prediction}")
            options = ["YES", "NO"]
            selected_option = st.selectbox("Is the prediction correct?", options)
            if selected_option == "NO":
                classes = [
                    "AbdomenCT",
                    "BreastMRI",
                    "ChestCT",
                    "CXR",
                    "Hand",
                    "HeadCT",
                ]  # Define your classes
                corrected_class = st.selectbox("Select Correct Class", classes)
                if st.button("Submit"):
                    st.write(f"Corrected Class: {corrected_class}")


if __name__ == "__main__":
    aws = AwsControl(config["aws_key"], config["aws_secret"])
    main()
