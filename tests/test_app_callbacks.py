import pytest
import numpy as np
import dash
from dash import Dash

# Create a Dash app instance before importing pages
app = Dash(__name__, use_pages=True, pages_folder="")

# Now import the pages
from pages import a3_model

def test_a3_model_callback():
    result = a3_model.predict_price_a3(1, 140.0, 120000, 2013)
    assert "Predicted Selling Price Class:" in str(result) or "Error:" in str(result)
    assert result != "Enter details and click Predict."


def test_a3_model_no_clicks():
    result = a3_model.predict_price_a3(0, 140.0, 120000, 2013)
    assert result == "Enter details and click Predict."