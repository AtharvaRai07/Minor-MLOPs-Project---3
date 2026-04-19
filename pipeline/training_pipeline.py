import sys
from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessor
from src.model_trainer import ModelTrainer

from src.logger import logging
from src.exception import CustomException

from config.paths_config import *

class TrainingPipeline:
    def __init__(self):
        pass

    def train(self):
        try:
            logging.info("Starting the training pipeline")

            ### 1. Data Ingestion
            data_ingestion = DataIngestion(config=CONFIG_PATH)
            data_ingestion.download_csv_from_gcp()

            ### 2. Data Processing
            data_processor = DataProcessor(input_file_path=RAW_FILE_PATH)
            data_processor.run()

            ### 3. Model Training and Evaluation
            model_trainer = ModelTrainer(X_train_path=X_TRAIN_PATH, X_test_path=X_TEST_PATH, y_train_path=Y_TRAIN_PATH, y_test_path=Y_TEST_PATH)
            model_trainer.run()

            logging.info("Training pipeline completed successfully")

        except Exception as e:
            logging.error("Error while running the training pipeline")
            raise CustomException(e, sys)

if __name__ == "__main__":
    training_pipeline = TrainingPipeline()
    training_pipeline.train()
    