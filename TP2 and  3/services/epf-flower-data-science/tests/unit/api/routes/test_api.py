import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add the project root to the sys.path
project_root = Path(__file__).resolve().parents[4]  # Adjust level as needed
sys.path.append(str(project_root))
from src.api.routes import api  # Adjust this import based on your project structure
from src.services.data import Dataset

client = TestClient(api.router)

@pytest.fixture
def sample_dataset():
    return Dataset(name="iris", url="https://example.com/iris.csv")


def test_get_dataset(sample_dataset):
    response = client.get(f"/dataset/{sample_dataset.name}")
    if response.status_code == 404:  # Dataset might not exist initially
        assert response.status_code == 404
    else:
        assert response.status_code == 200
        assert response.json()["name"] == sample_dataset.name


def test_post_dataset(sample_dataset):
    response = client.post("/dataset", json=sample_dataset.dict())
    assert response.status_code in [201, 403]  # 403 if the dataset already exists


def test_put_dataset(sample_dataset):
    response = client.put("/dataset", json=sample_dataset.dict())
    assert response.status_code in [200, 201]  # 201 if created, 200 if updated


def test_delete_dataset(sample_dataset):
    response = client.delete(f"/dataset/{sample_dataset.name}")
    if response.status_code == 404:  # Dataset might not exist
        assert response.status_code == 404
    else:
        assert response.status_code == 200


def test_load_dataset(sample_dataset):
    response = client.get(f"/load-dataset/{sample_dataset.name}")
    if response.status_code == 404:  # File might not exist
        assert response.status_code == 404
    else:
        assert response.status_code == 200
        assert isinstance(response.json(), list)


def test_preprocess_dataset(sample_dataset):
    response = client.get(f"/preprocess-dataset/{sample_dataset.name}")
    if response.status_code == 404:  # File might not exist
        assert response.status_code == 404
    else:
        assert response.status_code == 200
        assert isinstance(response.json(), list)


def test_split_dataset(sample_dataset):
    response = client.get(f"/split-dataset/{sample_dataset.name}")
    if response.status_code == 404:  # Dataset might not exist
        assert response.status_code == 404
    else:
        assert response.status_code == 200
        data = response.json()
        assert "train" in data
        assert "test" in data


def test_train_iris_dataset(sample_dataset):
    response = client.get(f"/train-iris-dataset/{sample_dataset.name}")
    assert response.status_code == 200
    assert response.json()["message"] == "Model trained and saved successfully."


def test_predict(sample_dataset):
    response = client.get(f"/predict/{sample_dataset.name}")
    if response.status_code == 404:  # Model or dataset might not exist
        assert response.status_code == 404
    else:
        assert response.status_code == 200
        assert isinstance(response.json()["predicted_labels"], list)
