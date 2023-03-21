# config.py
import os

list_of_files = list(os.walk(os.getcwd()))
root = list_of_files[0][0]
TRAIN_TEST_FILE_PATH = os.path.join(root, "input")
MODEL_OUTPUT_PATH = os.path.join(root, "models")