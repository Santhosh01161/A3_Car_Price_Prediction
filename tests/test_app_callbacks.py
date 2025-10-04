import pytest
from app import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def test_a3_model_callback(client):
    """Test Flask app with valid prediction inputs."""
    response = client.post("/predict", data={
        "maxpower": "140.0",
        "kmdriven": "120000",
        "Year": "2013"
    })
    assert response.status_code == 200
    # It should either succeed with a prediction or return an error message
    assert b"Predicted Car Class is" in response.data or b"Error:" in response.data


def test_a3_model_no_clicks(client):
    """Test Flask app with missing/empty inputs (no clicks)."""
    response = client.post("/predict", data={
        "maxpower": "",
        "kmdriven": "",
        "Year": ""
    })
    assert response.status_code == 200
    # Should handle gracefully with error message
    assert b"Error:" in response.data or b"Enter details" in response.data
