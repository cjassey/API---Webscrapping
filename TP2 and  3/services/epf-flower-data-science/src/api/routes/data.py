from fastapi.responses import JSONResponse
from src.services.data import Dataset, get_dataset_infos, open_configs_file, write_configs_file, dump_configs_file, load_dataset, process_dataset, split_dataset, train_dataset, predict_iris
from fastapi import APIRouter, HTTPException, status
import pandas as pd
from requests.exceptions import HTTPError
import requests
import zipfile
import io
from pathlib import Path
import json

router = APIRouter()


@router.get("/dataset/{dataset_id}", response_model=Dataset)
async def get_dataset(dataset_id: str):
    """ Get the information of a dataset from the configuration file

    Args:
        dataset_id (str): The name of the dataset to get

    Returns:
        Dataset: The dataset information

    Raises:
        404: The dataset was not found
    """
    dataset: Dataset = get_dataset_infos(dataset_id)
    return dataset


@router.post("/dataset")
async def post_dataset(dataset: Dataset):
    """ Add a new dataset to the configuration file

    Args:
        dataset (Dataset): The dataset to add

    Returns:
        Dataset: The dataset that was added

    Raises:
        201: The dataset was successfully added
        403: The dataset already exists
        500: The configuration file was not found / Error happened while reading it
    """
    urls = open_configs_file()
    if dataset.name in urls:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Dataset already exists: {dataset.name}. Use PUT if you wish to update it.")
    write_configs_file(dataset)
    return JSONResponse(
        content=get_dataset_infos(dataset.name).dict(),
        status_code=status.HTTP_201_CREATED
    )


@router.put("/dataset", response_model=Dataset)
async def put_dataset(dataset: Dataset):
    """ Update an existing dataset in the configuration file.
        If the dataset does not exist, it will be created and a 201 status code will be returned.

    Args:
        dataset (Dataset): The dataset to update

    Returns:
        Dataset: The updated dataset information

    Raises:
        200: The dataset was successfully updated
        201: The dataset was successfully added
        500: The configuration file was not found / Error happened while reading it / Error happened while writing it
    """
    urls = open_configs_file()
    ressource_exists = dataset.name in urls

    write_configs_file(dataset)

    updated_dataset_info = get_dataset_infos(dataset.name)
    if ressource_exists:
        return JSONResponse(
            content=updated_dataset_info.dict(),
            status_code=status.HTTP_200_OK
        )
    else:
        return JSONResponse(
            content=updated_dataset_info.dict(),
            status_code=status.HTTP_201_CREATED
        )


@router.delete("/dataset/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """ Delete a dataset from the configuration file

    Args:
        dataset_id (str): The name of the dataset to delete

    Returns:
        204: The dataset was successfully deleted

    Raises:
        404: The dataset was not found
        500: The configuration file was not found / Error happened while reading it / Error happened while writing it
    """
    datasets = open_configs_file()
    if dataset_id not in datasets:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset not found: {dataset_id}")
    del datasets[dataset_id]
    dump_configs_file(datasets)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": f"Dataset {dataset_id} was successfully deleted"}
    )

@router.get("/load-dataset/{dataset_name}")
async def load_dataset(dataset_name: str):
    """Load the dataset file as a DataFrame and return it as JSON.
    
     Args:
        dataset_name (str): The name of the file to load

    Returns:
        200: The file was successfully loaded

    Raises:
        404: The file was not found

    
    """
    try:
        df = load_dataset(dataset_name)
        return JSONResponse(content=df.to_dict(orient="records"))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"The file {dataset_name}.csv does not exist.")
    
@router.get("/preprocess-dataset/{dataset_name}")
async def preprocess_dataset(dataset_name: str):
    """processe the dataset file as a DataFrame and return it as JSON.
    
     Args:
        dataset_name (str): The name of the dataset to load and to process

    Returns:
        200: The dataset was successfully process

    Raises:
        404: The dataset was not found

    
    """
    try:
        df = process_dataset(dataset_name)
        return JSONResponse(content=df.to_dict(orient="records"))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"The file {dataset_name}.csv does not exist.")
    
@router.get("/split-dataset/{dataset_name}")
async def split_iris_dataset_endpoint(dataset_name: str):
    """
    Loads, processes, and splits a specified dataset into training (train) 
    and testing (test) sets. Returns the splits in JSON format.

    Arguments:
    - dataset_name (str): The name of the dataset to load and split.

    Returns:
    - JSON containing two keys:
      - "train": Training data as a list of dictionaries.
      - "test": Testing data as a list of dictionaries.
    """
    X_train, X_test, y_train, y_test = split_dataset(dataset_name)
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    response_data = {
        "train": train_df.to_dict(orient="records"),
        "test": test_df.to_dict(orient="records")
    }

    return JSONResponse(content=json.dumps(response_data, default=str))


@router.get("/train-iris-dataset/{dataset_name}")
async def train_iris_dataset_endpoint(dataset_name: str):
    """
    Trains a model on a specified dataset and saves the model.

    Arguments:
    - dataset_name (str): The name of the dataset to use for training.

    Returns:
    - JSON containing a message confirming that the model was trained 
      and saved successfully.
    """
    model = train_dataset(dataset_name)
    return JSONResponse(content={"message": "Model trained and saved successfully."})

    

@router.get('/predict/{dataset_name}')
async def predict(dataset_name: str):
    """
    Predicts labels for a specified dataset using a pre-trained model.

    Arguments:
    - dataset_name (str): The name of the dataset to use for predictions.

    Returns:
    - JSON containing the predicted labels as a list.
    """
    predicted_labels = predict_iris(dataset_name="iris")
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "predicted_labels": predicted_labels.tolist()
        }
        
    )


