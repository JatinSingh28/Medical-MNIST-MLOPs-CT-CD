import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from aws_controls import AwsControl
from config import config
from ResNetDir import ResNet
import mlflow
from mlflow import MlflowClient
import dagshub


# Define a function to make predictions
def predict(image):
    class_dict = {
        0: "AbdomenCT",
        1: "BreastMRI",
        2: "ChestCT",
        3: "CXR",
        4: "Hand",
        5: "HeadCT",
    }
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]
    )
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    # Perform prediction
    with torch.no_grad():
        outputs = st.session_state["model"](image_tensor)
        _, predicted = torch.max(outputs, 1)
        return class_dict[predicted.item()]


# Streamlit app
def main():
    if "prediction" not in st.session_state:
        st.session_state["prediction"] = None
        st.session_state["uploaded"] = False

    model = None
    if "loaded" not in st.session_state:
        if 'aws' not in st.session_state:
            st.session_state['aws'] = AwsControl(st.secrets["aws_key"], st.secrets["aws_secret"])
        dagshub.init("Medical-MNIST-MLOPs-CT-CD", "JatinSingh28", mlflow=True)
        client = MlflowClient()
        registered_model_name = "Production Model V1"
        version = client.get_model_version_by_alias(
            registered_model_name, "prod"
        ).version

        model_uri = f"models:/{registered_model_name}/{version}"

        if "model" not in st.session_state:
            st.session_state["model"] = mlflow.pytorch.load_model(model_uri)
            
        # model.load_state_dict(torch.load("./ResNet/model_ckpt/last-v4.ckpt")["state_dict"])
        st.session_state["model"].eval()
        st.session_state["loaded"] = True

    st.title("Image Classification")

    # with st.expander("Download sample images"):

    #     st.download_button(
    #         "Download",
    #         data=sample_imgs,
    #         file_name="sample_imgs.zip",
    #         mime="application/zip",
    #     )
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is None:
        st.session_state["prediction"] = None
        st.session_state["uploaded"] = False

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        image.save("uploaded_img.jpeg")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Classify"):
            st.session_state["prediction"] = predict(image)
            prediction = st.session_state["prediction"]
            st.success(f"Prediction Class: {prediction}")

        # st.session_state
        if st.session_state["prediction"] is not None:
            options = ["YES", "NO"]
            st.write("Select NO for queuing image for retraining")
            selected_option = st.radio("Is the prediction correct?", options)
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
                    if not st.session_state["uploaded"]:
                        st.session_state['aws'].upload_image("uploaded_img.jpeg", corrected_class)
                        st.success(
                            f"Image and corrected class {corrected_class} uploaded to AWS for retraining"
                        )
                        st.session_state["uploaded"] = True
                    else:
                        st.warning("Image already uploaded for retraining")


if __name__ == "__main__":
    # aws = AwsControl(config["aws_key"], config["aws_secret"])
    main()
