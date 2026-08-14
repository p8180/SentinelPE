import json
from pathlib import Path
import pytest

from src.api.app import app, load_model, expected_columns

@pytest.fixture()
def client():
    app.config.update(TESTING=True)
    with app.test_client() as client:
        yield client

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["feature_count"] > 0

def test_home(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"SentinelPE" in response.data

def test_demo_samples_exist():
    path = Path(__file__).resolve().parents[1] / "data" / "demo_samples.json"
    assert path.exists(), "Run scripts/export_demo_samples.py first."
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "goodware" in data and "malware" in data

def test_demo_predictions(client):
    path = Path(__file__).resolve().parents[1] / "data" / "demo_samples.json"
    if not path.exists():
        pytest.skip("demo_samples.json not exported yet")
    data = json.loads(path.read_text(encoding="utf-8"))

    good = client.post("/predict", json={"features": data["goodware"]["features"]})
    mal = client.post("/predict", json={"features": data["malware"]["features"]})

    assert good.status_code == 200
    assert mal.status_code == 200
    assert good.get_json()["label"] == "goodware"
    assert mal.get_json()["label"] == "malware"

def test_invalid_predict(client):
    response = client.post("/predict", json={"features": {}})
    assert response.status_code == 400
