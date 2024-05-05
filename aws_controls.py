import boto3
import logging
import os

# from dagshub import get_repo_bucket_client
# from io import StringIO
# from botocore.exceptions import NoCredentialsError
# import pandas as pd


class AwsControl:
    def __init__(
        self,
        aws_key,
        aws_secret,
        bucket_name="medical-mnist-mlops",
        region_name="ap-south-1",
    ):
        # self.s3 = get_repo_bucket_client(
        #     "JatinSingh28/Medical-Image-Classification", flavour="boto"
        # )
        self.aws_key = aws_key
        self.aws_secret = aws_secret
        self.region_name = region_name
        self.bucket_name = bucket_name

    def upload_image(self, image_path, class_name):
        s3 = boto3.client(
            "s3",
            region_name=self.region_name,
            aws_access_key_id=self.aws_key,
            aws_secret_access_key=self.aws_secret,
        )
        response = s3.list_objects_v2(Bucket=self.bucket_name, Prefix=class_name)
        file_count = len(response.get("Contents", []))
        s3.upload_file(
            image_path,
            self.bucket_name,
            f"{class_name}/image_{file_count}.{image_path.split('.')[-1]}",
        )
        logging.info(f"Successfully uploaded {image_path} to {self.bucket_name}")

    def download_s3_bucket(self, local_dir="data", bucket_name="medical-mnist-mlops"):
        s3 = boto3.client(
            "s3",
            region_name=self.region_name,
            aws_access_key_id=self.aws_key,
            aws_secret_access_key=self.aws_secret,
        )

        # List all objects in the bucket
        paginator = s3.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=bucket_name)

        for page in page_iterator:
            if "Contents" in page:
                for obj in page["Contents"]:
                    key = obj["Key"]
                    local_file = os.path.join(local_dir, key)

                    os.makedirs(os.path.dirname(local_file), exist_ok=True)

                    s3.download_file(bucket_name, key, local_file)
                    print(f"Downloaded: {key}")

    def is_image_file(self, file_path):
        # List of image file extensions
        image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]

        # Get the file extension
        _, file_extension = os.path.splitext(file_path.lower())

        # Check if the file extension is in the list of image extensions
        return file_extension in image_extensions

    def delete_s3_folder_images(
        self, local_dir="data", bucket_name="medical-mnist-mlops"
    ):
        s3 = boto3.client(
            "s3",
            region_name=self.region_name,
            aws_access_key_id=self.aws_key,
            aws_secret_access_key=self.aws_secret,
        )

        # List all objects in the bucket
        paginator = s3.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=bucket_name)

        for page in page_iterator:
            if "Contents" in page:
                for obj in page["Contents"]:
                    key = obj["Key"]
                    local_file = os.path.join(local_dir, key)

                    # Check if the key represents an image file
                    if os.path.isfile(local_file) and self.is_image_file(key):
                        # Delete the image file from the S3 bucket
                        s3.delete_object(Bucket=bucket_name, Key=key)
                        print(f"Deleted: {key}")

    # def upload_to_s3(self, df, file_name, bucket_name):
    #     """Uploads a file to AWS S3."""
    #     if file_name is None or df is None:
    #         logging.error("No file or data to upload.")
    #         return

    #     try:
    #         s3 = boto3.client(
    #             "s3",
    #             region_name=self.region_name,
    #             aws_access_key_id=self.aws_key,
    #             aws_secret_access_key=self.aws_secret,
    #         )

    #         csv_buffer = df.to_csv(index=False).encode()
    #         s3.put_object(Bucket=bucket_name, Key=file_name, Body=csv_buffer)
    #         logging.info(f"Successfully uploaded {file_name} to {bucket_name}")
    #     except FileNotFoundError:
    #         logging.error(f"The file {file_name} was not found")
    #     except NoCredentialsError:
    #         logging.error("Credentials not available.")
    #     except Exception as e:
    #         logging.error(f"An unknown error occurred: {e}")

    # def download_from_s3(self, bucket_name, file_name):
    #     """Downloads a file from AWS S3."""
    #     try:
    #         s3 = boto3.client(
    #             "s3",
    #             region_name=self.region_name,
    #             aws_access_key_id=self.aws_key,
    #             aws_secret_access_key=self.aws_secret,
    #         )

    #         csv_obj = s3.get_object(Bucket=bucket_name, Key=file_name)
    #         body = csv_obj["Body"].read().decode("utf-8")
    #         df = pd.read_csv(StringIO(body))
    #         logging.info(f"Successfully downloaded {file_name} from {bucket_name}.")
    #         return df
    #     except Exception as e:
    #         logging.error(f"An unknown error occurred.")
    #         raise e

    # def delete_csv(self, bucket, file_name):
    #     try:
    #         s3 = boto3.client(
    #             "s3",
    #             region_name=self.region_name,
    #             aws_access_key_id=self.aws_key,
    #             aws_secret_access_key=self.aws_secret,
    #         )
    #         s3.delete_object(Bucket=bucket, Key=file_name)
    #         logging.info(f"Deleted {file_name} from S3.")
    #     except Exception as e:
    #         logging.error(
    #             f"Error occurred: {e}. Unable to locate file or file not present"
    #         )
    #         raise e
