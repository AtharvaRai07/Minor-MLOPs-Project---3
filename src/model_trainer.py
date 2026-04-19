import os
import sys
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from config.paths_config import *
from config.model_params import *
from utils.common_functions import read_csv
from src.logger import logging
from src.exception import CustomException

class ModelTrainer:
    def __init__(self, X_train_path: str, X_test_path: str, y_train_path: str, y_test_path: str):
        self.X_train = read_csv(X_train_path)
        self.X_test = read_csv(X_test_path)
        self.y_train = read_csv(y_train_path)
        self.y_test = read_csv(y_test_path)

        self.model = None

        logging.info("Training and testing data loaded successfully")

        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(MODEL_EVALUTAION_DIR, exist_ok=True)

    def train_model(self):
        try:
            logging.info("Training the Decision Tree Classifier model")

            self.model = DecisionTreeClassifier(**MODEL_PARAMS)
            self.model.fit(self.X_train, self.y_train.values.ravel())

            joblib.dump(self.model, MODEL_FILE_PATH)

            logging.info("Model trained successfully")

        except Exception as e:
            logging.error("Error while training the model")
            raise CustomException(e, sys)

    def evaluate_model(self):
        try:
            logging.info("Evaluating the model on the test set")

            y_pred = self.model.predict(self.X_test)

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')

            classification_rep = classification_report(self.y_test, y_pred)
            conf_matrix = confusion_matrix(self.y_test, y_pred)

            logging.info(f"Model evaluation metrics - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

            with open(MODEL_SCORE_PATH, 'w') as f:
                f.write(f"Accuracy: {accuracy}\n")
                f.write(f"Precision: {precision}\n")
                f.write(f"Recall: {recall}\n")
                f.write(f"F1 Score: {f1}\n")

            logging.info("Model evaluation metrics saved successfully")

            with open(CLASSIFICATION_REPORT_PATH, 'w') as f:
                f.write(classification_rep)

            logging.info("Classification report saved successfully")

            logging.info("Saving confusion matrix and decision tree plot")

            plt.figure(figsize=(10, 7))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(self.y_test), yticklabels=np.unique(self.y_test))
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.savefig(CONFUSION_MATRIX_PATH)
            plt.close()

            plt.figure(figsize=(10, 7))
            plot_tree(self.model, feature_names=self.X_train.columns, filled=True)
            plt.savefig(DECISION_TREE_PLOT_PATH)
            plt.close()

            logging.info("Model evaluation completed successfully")
        except Exception as e:
            logging.error("Error while evaluating the model")
            raise CustomException(e, sys)

    def run(self):
        try:
            logging.info("Starting the model training and evaluation process")

            self.train_model()
            self.evaluate_model()

            logging.info("Model training and evaluation process completed successfully")

        except Exception as e:
            logging.error("Error while running the model training and evaluation process")
            raise CustomException(e, sys)


if __name__ == "__main__":
    trainer = ModelTrainer(X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH)
    trainer.run()
