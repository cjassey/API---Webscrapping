from http.client import InvalidURL
import io
import zipfile
from fastapi import HTTPException, status
from pydantic import BaseModel, validator
import requests
from pathlib import Path
import json
import validators
import pandas as pd
from requests.exceptions import HTTPError

JSON_CONFIG_PATH = Path(__file__).parent.parent / "config/urls_config.json"
OUTPUT_FILE_PATH = Path(__file__).parent.parent / "data"


class Dataset(BaseModel):
    name: str
    url: str

    @validator("url")
    def url_must_be_valid(cls, url: str) -> str:
        if not validators.url(url):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid URL: {url}"
            )
        return url

    @validator("name")
    def name_must_be_valid(cls, name: str) -> str:
        if not name.isalnum():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid name: {name}"
            )
        if not isinstance(name, str):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Name must be a string: {name}"
            )
        return name


def open_configs_file() -> dict:
    """ Open the configuration file and return its content as a dictionary """
    try:
        with open(JSON_CONFIG_PATH) as file:
            datasets = json.load(file)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration file not found: {JSON_CONFIG_PATH}")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while reading the configuration file: {e}")
    else:
        return datasets


def dump_configs_file(datasets: dict) -> None:
    """ Dump the datasets dictionary to the configuration file """
    try:
        with open(JSON_CONFIG_PATH, "w") as file:
            json.dump(datasets, file, indent=4)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while writing the configuration file {e}")


def write_configs_file(dataset: Dataset) -> None:
    """ Take a dataset and write it to the configuration file """

    # We put this line out of the try block since the exceptions are already
    # handled in the open_configs_file function
    datasets = open_configs_file()
    try:
        datasets[dataset.name] = dataset.dict()
        with open(JSON_CONFIG_PATH, "w") as file:
            json.dump(datasets, file, indent=4)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while writing the configuration file {e}")


def get_dataset_infos(dataset_id: str) -> Dataset:
    """ Get the information of a dataset from the configuration file """
    try:
        datasets = open_configs_file()
        return Dataset(**datasets[dataset_id])
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"Dataset not found in configuration file: {dataset_id}")


def load_iris_dataset(dataset_name: str) -> pd.DataFrame:
    file_path =  f"src/data/{dataset_name}.csv"
    print(file_path)
    return pd.read_csv(f"{file_path}")
    
    
def process_iris_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Effectuer le traitement nécessaire sur le dataset Iris."""
    # Exemple de traitement : suppression de la colonne 'Id'
    df = df.drop(columns=['Id'])
    # Ajouter d'autres traitements si nécessaire
    return df
