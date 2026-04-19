import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config.paths_config import *
from utils.common_functions import read_csv, save_csv

from src.logger import logging
from src.exception import CustomException

class DataProcessor:
    def __init__(self, input_file_path):
        self.file_path = input_file_path
        self.df = None
        self.X = None
        self.y = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        os.makedirs(PREPROCESSING_DIR, exist_ok=True)

    def load_data(self):
        try:
            logging.info(f"Loading data from {self.file_path}")

            self.df = read_csv(self.file_path)

            logging.info("Data loaded successfully")

        except Exception as e:
            logging.error("Error while loading the data")
            raise CustomException(e, sys)

    def handling_outliers(self):
        try:
            logging.info("Handling outliers in the SepalWidthCm using IQR method")

            Q1 = self.df['SepalWidthCm'].quantile(0.25)
            Q3 = self.df['SepalWidthCm'].quantile(0.75)

            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            median_value = self.df['SepalWidthCm'].median()

            self.df['SepalWidthCm'] = self.df['SepalWidthCm'].apply(lambda x: median_value if x < lower_bound or x > upper_bound else x)
            logging.info("Outliers handled successfully")

        except Exception as e:
            logging.error("Error while handling outliers in the data")
            raise CustomException(e, sys)

    def remove_unwanted_columns(self):
        try:
            logging.info("Removing Id column from data")

            self.df.drop('Id', axis=1, inplace=True)

            logging.info("Id column removed successfully")

        except Exception as e:
            logging.error("Error while removing unwanted columns from the data")
            raise CustomException(e, sys)

    def split_data(self):
        try:
            logging.info("Splitting data into features and target variable")

            self.X = self.df.drop('Species', axis=1)
            self.y = self.df['Species']

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)

            logging.info("Data split into features and target variable successfully")

        except Exception as e:
            logging.error("Error while splitting the data")
            raise CustomException(e, sys)

    def encode_target_variable(self):
        try:
            logging.info("Encoding target variable using LabelEncoder")

            le = LabelEncoder()
            self.y_train = le.fit_transform(self.y_train)
            self.y_test = le.transform(self.y_test)

            os.makedirs(LABEL_ENCODER_DIR, exist_ok=True)
            joblib.dump(le, LABEL_ENCODER_PATH)

            logging.info("Target variable encoded successfully")
            logging.info(f"LabelEncoder saved at {LABEL_ENCODER_PATH}")

        except Exception as e:
            logging.error("Error while encoding the target variable")
            raise CustomException(e, sys)

    def scale_data(self):
        try:
            logging.info("Scaling features using StandardScaler")

            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)

            os.makedirs(SCALER_DIR, exist_ok=True)
            joblib.dump(scaler, SCALER_PATH)

            logging.info("Features scaled successfully")
            logging.info(f"StandardScaler saved at {SCALER_PATH}")

        except Exception as e:
            logging.error("Error while scaling the features")
            raise CustomException(e, sys)

    def run(self):
        try:
            logging.info("Running the data processing pipeline")

            self.load_data()
            self.handling_outliers()
            self.remove_unwanted_columns()
            self.split_data()
            self.encode_target_variable()
            self.scale_data()

            save_csv(self.X_train, X_TRAIN_PATH)
            save_csv(self.X_test, X_TEST_PATH)
            save_csv(self.y_train, Y_TRAIN_PATH)
            save_csv(self.y_test, Y_TEST_PATH)

            logging.info("Data processing pipeline completed successfully")

        except Exception as e:
            logging.error("Error while running the data processing pipeline")
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_processor = DataProcessor(input_file_path=RAW_FILE_PATH)
    data_processor.run()
