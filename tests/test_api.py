import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# These tests mock the ML models so they don't need GPU or trained weights


class TestAPIEndpoints:
    """Test API endpoints with mocked predictor."""
    
    @pytest.fixture
    def client(self):
        """Create test client with mocked predictor."""
        from fastapi.testclient import TestClient
        
        # Mock the predictor
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = {
            'class': 'Tumor',
            'label': 1,
            'confidence': 0.95,
            'probability': 0.95
        }
        mock_predictor.model = MagicMock()
        mock_predictor.model.name = 'mock_model'
        mock_predictor.model.input_shape = (None, 224, 224, 3)
        mock_predictor.model.count_params.return_value = 11137
        # Count trainable params
        mock_predictor.model.trainable_weights = []
        mock_predictor.model.non_trainable_weights = []
        
        # Patch load_trained_model to avoid loading real model
        with patch('brain_tumor_detection.api.app.Predictor') as MockPredictor, \
             patch('brain_tumor_detection.api.app.GradCAM') as MockGradCAM:
            MockPredictor.return_value = mock_predictor
            MockGradCAM.return_value = MagicMock()
            
            from brain_tumor_detection.api.app import create_app
            app = create_app(model_path='fake_model.keras')
            with TestClient(app) as test_client:
                yield test_client
    
    def test_health_check(self, client):
        response = client.get('/health')
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
    
    def test_model_info(self, client):
        response = client.get('/model/info')
        assert response.status_code == 200
    
    def test_predict_no_file(self, client):
        response = client.post('/predict')
        assert response.status_code == 422  # Validation error
    
    def test_predict_invalid_file_type(self, client):
        """Should reject non-image files."""
        from io import BytesIO
        fake_file = BytesIO(b'not an image')
        response = client.post('/predict', files={'file': ('test.txt', fake_file, 'text/plain')})
        assert response.status_code in [400, 422]
