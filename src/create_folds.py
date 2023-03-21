import pandas as pd
from sklearn import model_selection
import config
import os


def train_folds(data, n_folds):

    # read the training data from csv file
    del_prev_files("input", "_folds")
    df = pd.read_csv(f"{config.TRAIN_TEST_FILE_PATH}\\{data}")

    # create a new column called kfold and fill it with -1
    df["kfold"] = -1

    # randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    # instantiate the kfold clas model from model_selection
    kf = model_selection.KFold(n_splits=n_folds)

    # fill the new kfold column
    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, "kfold"] = fold

    # save the new file with kfold column
    index = data.find(".")
    filename = data[:index]
    filename = f"{filename}_folds.csv"
    df.to_csv(f"{config.TRAIN_TEST_FILE_PATH}\\{filename}", index=False)


def del_prev_files(folder, starts):
    """ "
    Delete previous reservoir model files before the start of optimization

    Parameters:
    folder (str): Name of folder with files to be deleted
    starts (str): Start str pattern of files to be deleted

    Returns:
    None"""
    list_of_files = list(os.walk(os.getcwd()))
    root = list_of_files[0][0]
    folder_path = os.path.join(root, folder)

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(f"{starts}.csv"):
            try:
                file_path = os.path.join(folder_path, filename)
                os.remove(file_path)
            except PermissionError as error:
                print(f"Failed to delete {file_path}. Reason: {error}")
            except IsADirectoryError as error:
                print(f"Failed to delete {file_path}. Reason: {error}")
