import opendatasets as od
import config
import os
import shutil

def download_kaggle_data(url: str = config.DATASET_URL) -> None:
    name = od.download(url)
    shutil.move("./raf-db-dataset", "./data")


if __name__ == "__main__":
    download_kaggle_data()