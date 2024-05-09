# Project Name: Medical MNIST Image Classification with MLFlow, Streamlit, Airflow, and AWS S3

## Description:
This project is designed to facilitate the classification of medical MNIST images using machine learning models. It employs a Streamlit frontend for user interaction, MLFlow for model management and experiment tracking, Airflow for automated data fetching, model retraining, and AWS S3 for storage of images.

## Pipeline Architecture:

![Pipeline Architecture](image_url)

1. **User Interaction:**
   - The pipeline begins with a Streamlit frontend, providing users with an intuitive interface to upload medical MNIST images for classification.

2. **Model Inference:**
   - Upon image upload, the pipeline fetches the latest trained model from the MLFlow model registry. This model is then used to make predictions on the uploaded images.

3. **Error Correction Mechanism:**
   - In case the prediction is incorrect, users have the option to provide the correct class label. If provided, the image along with the corrected label is stored in AWS S3 for future reference.

4. **Airflow DAG:**
   - Scheduled to run weekly, an Airflow Directed Acyclic Graph (DAG) automates the process of fetching new data from AWS S3 and retrieving the latest model from the MLFlow model registry.

5. **Model Retraining:**
   - Once the new data and model are fetched, the pipeline initiates the process of retraining the model on the updated dataset. This ensures that the model remains up-to-date and capable of making accurate predictions.

6. **Model Registry Update:**
   - Upon successful retraining, the newly trained model is uploaded to the MLFlow model registry, replacing the previous version. This ensures that the latest model is readily available for inference in subsequent interactions with the Streamlit frontend.

7. **Continuous Improvement:**
   - The iterative nature of the pipeline ensures continuous improvement in model performance over time as it adapts to new data and updates.

## Features:
1. **Streamlit Frontend:** Users can upload medical MNIST images for classification.
2. **MLFlow Integration:** Models are fetched from the MLFlow model registry to make predictions.
3. **Correction Mechanism:** If the prediction is incorrect, users can provide the correct class label, and the image with the corrected class label will be uploaded to AWS S3.
4. **Airflow DAG:** Scheduled to run weekly, Airflow DAG fetches new data from AWS and the latest model from the MLFlow model registry, retrains the model on new data, and uploads the new model to the MLFlow registry.
5. **AWS S3 Integration:** Images with corrected labels are stored in AWS S3 for future reference and analysis.

## Components:
1. **Streamlit Frontend:** Handles user interaction and image uploads.
2. **MLFlow Model Registry:** Stores trained models and facilitates model deployment.
3. **Airflow DAG:** Automates data fetching, model retraining, and model deployment.
4. **AWS S3:** Stores images with corrected labels for future use.

## Setup Instructions:
1. Clone the repository from GitHub.
2. Install the necessary dependencies listed in the requirements.txt file.
3. Configure AWS credentials for S3 access.
4. Create env file to store MLFlow URI and Dags hub credentials.
5. Configure Airflow with appropriate connections and DAG configurations.
6. Run the Streamlit application.
7. Ensure Airflow scheduler is running to trigger DAG executions.

## Configuration Instructions

To run the project, please follow these configuration steps:

1. **Create a config.py File:**
    Create a `config.py` file in the root directory of the project with the following content:
    ```python
    config = {
        "aws_key": "YOUR_AWS_KEY",
        "aws_secret": "YOUR_AWS_SECRET",
    }
    ```

2. **Set Environment Variables:**
    Set the following environment variables in your environment:
    ```plaintext
    MLFLOW_TRACKING_URI = https://dagshub.com/JatinSingh28/Medical-Image-Classification.mlflow
    MLFLOW_TRACKING_USERNAME = JatinSingh28
    MLFLOW_TRACKING_PASSWORD = YOUR_MLFLOW_PASSWORD
    DAGSHUB_USER_TOKEN = YOUR_DAGSHUB_USER_TOKEN
    ```
    Replace `YOUR_AWS_KEY`, `YOUR_AWS_SECRET`, `YOUR_MLFLOW_PASSWORD`, and `YOUR_DAGSHUB_USER_TOKEN` with your actual credentials.

These configurations are necessary for proper functioning of the project.

## Usage:
1. Access the Streamlit frontend through the provided URL.
2. Upload medical MNIST images for classification.
3. View predictions and correct any inaccuracies if necessary.
4. The Airflow DAG will automatically fetch new data and retrain the model weekly.
5. Corrected images are stored in AWS S3 for future analysis.

## Contributing:
1. Fork the repository on GitHub.
2. Make necessary changes and improvements.
3. Create a pull request with detailed information about the changes.
4. Contributions will be reviewed and merged if deemed appropriate.

## Authors:
[List of project contributors]

## License:
This project is licensed under the [License Name] license. See the LICENSE.md file for details.

## Acknowledgements:
1. Mention any libraries, frameworks, or tools used in the project.
2. Acknowledge any individuals or organizations whose work or contributions influenced the project.

## Contact:
For inquiries or support, please contact [email address].