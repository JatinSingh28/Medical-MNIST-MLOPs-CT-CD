import boto3
import logging
from io import StringIO
from botocore.exceptions import NoCredentialsError
import pandas as pd


class AwsControl:
    def __init__(self, aws_key, aws_secret, region_name="ap-south-1"):
        self.aws_key = aws_key
        self.aws_secret = aws_secret
        self.region_name = region_name
        
    def upload_image(self,image_path,bucket_name,class_name):
        s3 = boto3.client(
            "s3",
            region_name=self.region_name,
            aws_access_key_id=self.aws_key,
            aws_secret_access_key=self.aws_secret,
        )
        s3.upload_file(image_path, bucket_name, f"{class_name}/{image_path.split('/')[-1]}")
        logging.info(f"Successfully uploaded {image_path} to {bucket_name}")

    def upload_to_s3(self, df, file_name, bucket_name):
        """Uploads a file to AWS S3."""
        if file_name is None or df is None:
            logging.error("No file or data to upload.")
            return

        try:
            s3 = boto3.client(
                "s3",
                region_name=self.region_name,
                aws_access_key_id=self.aws_key,
                aws_secret_access_key=self.aws_secret,
            )

            csv_buffer = df.to_csv(index=False).encode()
            s3.put_object(Bucket=bucket_name, Key=file_name, Body=csv_buffer)
            logging.info(f"Successfully uploaded {file_name} to {bucket_name}")
        except FileNotFoundError:
            logging.error(f"The file {file_name} was not found")
        except NoCredentialsError:
            logging.error("Credentials not available.")
        except Exception as e:
            logging.error(f"An unknown error occurred: {e}")

    def download_from_s3(self, bucket_name, file_name):
        """Downloads a file from AWS S3."""
        try:
            s3 = boto3.client(
                "s3",
                region_name=self.region_name,
                aws_access_key_id=self.aws_key,
                aws_secret_access_key=self.aws_secret,
            )

            csv_obj = s3.get_object(Bucket=bucket_name, Key=file_name)
            body = csv_obj["Body"].read().decode("utf-8")
            df = pd.read_csv(StringIO(body))
            logging.info(f"Successfully downloaded {file_name} from {bucket_name}.")
            return df
        except Exception as e:
            logging.error(f"An unknown error occurred.")
            raise e

    def delete_csv(self, bucket, file_name):
        try:
            s3 = boto3.client(
                "s3",
                region_name=self.region_name,
                aws_access_key_id=self.aws_key,
                aws_secret_access_key=self.aws_secret,
            )
            s3.delete_object(Bucket=bucket, Key=file_name)
            logging.info(f"Deleted {file_name} from S3.")
        except Exception as e:
            logging.error(
                f"Error occurred: {e}. Unable to locate file or file not present"
            )
            raise e
