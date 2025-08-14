from llama_stack_provider_trustyai_fms.config import (
    DetectorParams,
    BaseDetectorConfig,
    ContentDetectorConfig,
    ChatDetectorConfig,
    FMSSafetyProviderConfig,
    MessageType,
)

import pytest
import os
from typing import Dict, Any


@pytest.fixture
def sample_detector_params() -> Dict[str, Any]:
    """Sample detector parameters for testing"""
    return {
        "temperature": 0.7,
        "risk_name": "test_risk",
        "risk_definition": "A test risk for unit testing",
        "regex": ["test.*pattern"],
        "custom_param": "custom_value"
    }


@pytest.fixture
def sample_base_detector_config() -> Dict[str, Any]:
    """Sample base detector configuration for testing"""
    return {
        "detector_id": "test_detector",
        "confidence_threshold": 0.8,
        "detector_url": "https://api.example.com/detector",
        "orchestrator_url": "https://api.example.com/orchestrator",
        "auth_token": "test_token_123",
        "verify_ssl": True,
        "message_types": {"user", "completion"}
    }


@pytest.fixture
def sample_fms_config() -> Dict[str, Any]:
    """Sample FMS safety provider configuration for testing"""
    return {
        "orchestrator_url": "https://api.example.com/orchestrator",
        "auth_token": "provider_token_456",
        "shields": {
            "content_shield": {
                "type": "content",
                "detector_url": "https://api.example.com/content",
                "confidence_threshold": 0.7,
                "detector_params": {
                    "temperature": 0.5,
                    "risk_name": "content_risk"
                }
            },
            "chat_shield": {
                "type": "chat",
                "detector_url": "https://api.example.com/chat",
                "confidence_threshold": 0.9,
                "detectors": {
                    "sub_detector_1": {
                        "detector_params": {"param1": "value1"}
                    }
                }
            }
        }
    }


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for SSL and auth testing"""
    original_values = {}
    test_env = {
        'FMS_VERIFY_SSL': 'false',
        'FMS_SSL_CERT_PATH': '/path/to/cert.pem',
        'FMS_AUTH_TOKEN': 'env_token_789'
    }

    for key, value in test_env.items():
        original_values[key] = os.environ.get(key)
        os.environ[key] = value

    yield test_env

    for key, original_value in original_values.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value

class TestDetectorParams:
    def test_init_with_kwargs(self, sample_detector_params):
        params = DetectorParams(**sample_detector_params)
        
        assert params.temperature == 0.7
        assert params.risk_name == "test_risk"
        assert params.risk_definition == "A test risk for unit testing"
        assert params.regex == ["test.*pattern"]
        assert params.kwargs["custom_param"] == "custom_value"

    def test_detectors_handling(self, sample_detector_params):
        detectors_config = {
            "detector1": {
                "detector_params": {
                    "param1": "value1",
                    "param2": "value2"
                }
            },
            "detector2": {
                "detector_params": {
                    "param3": "value3"
                }
            }
        }
        
        detector_params = DetectorParams(**sample_detector_params)
        detector_params.detectors = detectors_config
        
        flattened = detector_params.orchestrator_detectors
        assert "detector1" in flattened
        assert "detector2" in flattened
        assert flattened["detector1"]["param1"] == "value1"
        assert flattened["detector1"]["param2"] == "value2"
        assert flattened["detector2"]["param3"] == "value3"

    def test_to_dict_methods(self, sample_detector_params):
        params = DetectorParams(**sample_detector_params)
        
        # Test to_dict (flattened)
        flat_dict = params.to_dict()
        assert "temperature" in flat_dict
        assert "risk_name" in flat_dict
        assert "custom_param" in flat_dict
        assert flat_dict["temperature"] == 0.7
        
        # Test to_categorized_dict
        categorized = params.to_categorized_dict()
        assert "model_params" in categorized
        assert "metadata" in categorized
        assert "kwargs" in categorized
        assert categorized["model_params"]["temperature"] == 0.7
        assert categorized["metadata"]["risk_name"] == "test_risk"


class TestBaseDetectorConfig:
    def test_init_basic(self, sample_base_detector_config):
        config = BaseDetectorConfig(**sample_base_detector_config)
        
        assert config.detector_id == "test_detector"
        assert config.confidence_threshold == 0.8
        assert config.detector_url == "https://api.example.com/detector"
        assert config.orchestrator_url == "https://api.example.com/orchestrator"
        assert config.auth_token == "test_token_123"
        assert config.verify_ssl is True
        assert config.message_types == {"user", "completion"}

    def test_init_with_defaults(self):
        config = BaseDetectorConfig(detector_id="minimal_detector")
        
        assert config.detector_id == "minimal_detector"
        assert config.confidence_threshold == 0.5
        assert config.message_types == MessageType.as_set()
        assert config.detector_params is not None
        assert isinstance(config.detector_params, DetectorParams)

    def test_message_types_validation(self):
        # Valid message types
        config = BaseDetectorConfig(
            detector_id="test",
            message_types=["user", "completion"]
        )
        assert config.message_types == {"user", "completion"}
        
        # Invalid message types
        with pytest.raises(ValueError):
            BaseDetectorConfig(
                detector_id="test",
                message_types=["invalid_type"]
            )

    def test_url_validation(self):
        # Valid URLs
        config = BaseDetectorConfig(
            detector_id="test",
            detector_url="https://api.example.com/detector"
        )
        config.validate()  # Should not raise
        
        # Invalid URL format
        with pytest.raises(ValueError, match="Invalid detector_url format"):
            config = BaseDetectorConfig(
                detector_id="test",
                detector_url="not-a-url"
            )
            config.validate()

    def test_ssl_config(self):
        config = BaseDetectorConfig(
            detector_id="test",
            verify_ssl=False,
            ssl_cert_path="/path/to/cert.pem",
            ssl_client_cert="/path/to/client.pem",
            ssl_client_key="/path/to/client.key"
        )
        
        ssl_config = config.get_ssl_config()
        assert ssl_config["verify"] is False
        assert ssl_config["cert"] == ("/path/to/client.pem", "/path/to/client.key")

    def test_auth_headers(self):
        config = BaseDetectorConfig(
            detector_id="test",
            auth_token="test_token_123"
        )
        
        headers = config.get_auth_headers()
        assert headers["Authorization"] == "Bearer test_token_123"
        
        # Test without token
        config.auth_token = None
        headers = config.get_auth_headers()
        assert headers == {}

    def test_use_orchestrator_api(self):
        config = BaseDetectorConfig(
            detector_id="test",
            orchestrator_url="https://api.example.com/orchestrator"
        )
        assert config.use_orchestrator_api is True
        
        config.orchestrator_url = None
        assert config.use_orchestrator_api is False

    def test_env_vars_ssl(self, mock_env_vars):
        config = BaseDetectorConfig(detector_id="test")
        
        # Should pick up env vars
        assert config.verify_ssl is False  # FMS_VERIFY_SSL=false
        assert config.ssl_cert_path == "/path/to/cert.pem"

class TestFMSSafetyProviderConfig:
    def test_init_basic(self, sample_fms_config):
        config = FMSSafetyProviderConfig(**sample_fms_config)
        
        assert config.orchestrator_url == "https://api.example.com/orchestrator"
        assert config.auth_token == "provider_token_456"
        
        # Check that shields were processed into detector configs
        assert "content_shield" in config.shields
        assert "chat_shield" in config.shields
        
        content_shield = config.shields["content_shield"]
        assert isinstance(content_shield, ContentDetectorConfig)
        assert content_shield.detector_id == "content_shield"
        assert content_shield.confidence_threshold == 0.7
        
        chat_shield = config.shields["chat_shield"]
        assert isinstance(chat_shield, ChatDetectorConfig)
        assert chat_shield.detector_id == "chat_shield"
        assert chat_shield.confidence_threshold == 0.9

    def test_detector_inheritance(self):
        config_data = {
            "orchestrator_url": "https://provider.example.com",
            "auth_token": "provider_token",
            "verify_ssl": False,
            "shields": {
                "test_shield": {
                    "type": "content",
                    "detector_url": "https://detector.example.com"
                }
            }
        }
        
        config = FMSSafetyProviderConfig(**config_data)
        shield = config.shields["test_shield"]
        
        # Should inherit provider values
        assert shield.orchestrator_url == "https://provider.example.com"
        assert shield.auth_token == "provider_token"
        assert shield.verify_ssl is False

    def test_get_detectors_by_type(self, sample_fms_config):
        config = FMSSafetyProviderConfig(**sample_fms_config)
        
        # Add message_types to sample config shields for testing
        config.shields["content_shield"].message_types = {"user"}
        config.shields["chat_shield"].message_types = {"user", "completion"}
        
        user_detectors = config.get_detectors_by_type(MessageType.USER)
        assert len(user_detectors) == 2
        assert "content_shield" in user_detectors
        assert "chat_shield" in user_detectors
        
        completion_detectors = config.get_detectors_by_type("completion")
        assert len(completion_detectors) == 1
        assert "chat_shield" in completion_detectors

    def test_nested_detector_params(self):
        config_data = {
            "shields": {
                "test_shield": {
                    "type": "content",
                    "detector_url": "https://detector.example.com",
                    "detector_params": {
                        "temperature": 0.8,
                        "risk_name": "test_risk"
                    },
                    "detectors": {
                        "sub_detector": {
                            "detector_params": {
                                "sub_param": "sub_value"
                            }
                        }
                    }
                }
            }
        }
        
        config = FMSSafetyProviderConfig(**config_data)
        shield = config.shields["test_shield"]
        
        # Check detector_params were processed
        assert shield.detector_params.temperature == 0.8
        assert shield.detector_params.risk_name == "test_risk"
        
        # Check nested detectors were preserved
        detectors = shield.detector_params.detectors
        assert detectors is not None
        assert "sub_detector" in detectors

    def test_env_var_inheritance(self, mock_env_vars):
        config_data = {
            "shields": {
                "test_shield": {
                    "type": "content",
                    "detector_url": "https://detector.example.com"
                }
            }
        }
        
        config = FMSSafetyProviderConfig(**config_data)
        shield = config.shields["test_shield"]
        
        # Should inherit env var values
        assert shield.verify_ssl is False  # FMS_VERIFY_SSL=false
        assert shield.ssl_cert_path == "/path/to/cert.pem"
        assert shield.auth_token == "env_token_789"