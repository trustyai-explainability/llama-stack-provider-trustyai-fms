import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from llama_stack.apis.inference import UserMessage
from llama_stack.apis.safety import RunShieldResponse

from llama_stack_provider_trustyai_fms.detectors.base import (
    DetectionResult,
    DetectorValidationError,
)
from llama_stack_provider_trustyai_fms.detectors.content import ContentDetector
from llama_stack_provider_trustyai_fms.detectors.chat import ChatDetector
from llama_stack_provider_trustyai_fms.config import (
    ContentDetectorConfig,
    ChatDetectorConfig,
    DetectorParams,
)


class TestContentDetector:
    @pytest.fixture
    def content_config(self):
        return ContentDetectorConfig(
            detector_id="test_content",
            detector_url="https://api.example.com/content",
            confidence_threshold=0.7,
            auth_token="test_token",
            detector_params=DetectorParams(
                temperature=0.5,
                risk_name="test_risk",
                regex=["test.*pattern"]
            )
        )

    @pytest.fixture
    def content_detector(self, content_config):
        return ContentDetector(content_config)
    
    @pytest.fixture
    def mock_shield_store(self, content_detector):
        mock_shield = MagicMock()
        mock_shield.identifier = "shield_id"
        mock_shield_store = AsyncMock()
        mock_shield_store.get_shield.return_value = mock_shield
        content_detector.shield_store = mock_shield_store
        return mock_shield_store

    def test_init(self, content_config):
        detector = ContentDetector(content_config)
        
        assert detector.config.detector_id == "test_content"
        assert detector.config.confidence_threshold == 0.7

    def test_init_invalid_config(self):
        chat_config = ChatDetectorConfig(detector_id="invalid chat config")
        
        with pytest.raises(DetectorValidationError):
            ContentDetector(chat_config)

    def test_extract_detector_params(self, content_detector):
        params = content_detector._extract_detector_params()
        
        assert params["temperature"] == 0.5
        assert params["risk_name"] == "test_risk"
        assert params["regex"] == ["test.*pattern"]

    @pytest.mark.asyncio
    async def test_detect_content(self, content_detector, mock_shield_store):
        mock_response_data = [
            {
                "detections": [
                    {
                        "score": 0.8,
                        "label": "label",
                        "detection_type": "content"
                    }
                ]
            }
        ]
        
        with patch.object(content_detector, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response_data
            
            messages = [UserMessage(content="Test message", role="user")]
            result = await content_detector.run_shield(shield_id=content_detector.config.detector_id, messages=messages)
            
            assert isinstance(result, RunShieldResponse)
            mock_request.assert_called_once()


    def test_prepare_content_request(self, content_detector):
        content = "Hello world"
        
        request_data = content_detector._prepare_content_request(content)
        
        assert "contents" in request_data
        assert len(request_data["contents"]) == 1
        assert request_data["contents"][0] == "Hello world"
        assert "detector_params" in request_data
        assert request_data["detector_params"]["temperature"] == 0.5


class TestChatDetector:
    @pytest.fixture
    def chat_config(self):
        return ChatDetectorConfig(
            detector_id="test_chat",
            detector_url="https://api.example.com/chat",
            orchestrator_url="https://api.example.com/orchestrator",
            confidence_threshold=0.8,
            detector_params=DetectorParams(
                temperature=0.7,
                detectors={
                    "detector_1": {
                        "detector_params": {
                            "param1": "value1"
                        }
                    }
                }
            )
        )

    @pytest.fixture
    def chat_detector(self, chat_config):
        return ChatDetector(chat_config)
    
    @pytest.fixture
    def mock_chat_shield_store(self, chat_detector):
        mock_shield = MagicMock()
        mock_shield.identifier = chat_detector.config.detector_id
        mock_shield_store = AsyncMock()
        mock_shield_store.get_shield.return_value = mock_shield
        chat_detector.shield_store = mock_shield_store
        return mock_shield_store

    def test_init(self, chat_config):
        detector = ChatDetector(chat_config)
        
        assert detector.config.detector_id == "test_chat"
        assert detector.config.confidence_threshold == 0.8
        assert isinstance(detector.config, ChatDetectorConfig)

    def test_init_invalid_config(self):
        content_config = ContentDetectorConfig(detector_id="test")
        
        with pytest.raises(DetectorValidationError, match="Config must be an instance of ChatDetectorConfig"):
            ChatDetector(content_config)

    @pytest.mark.asyncio
    async def test_detect_chat_orchestrator_mode(self, chat_detector, mock_chat_shield_store):
        mock_response_data = {
            "detections": [
                {
                    "detector_id": "sub_detector_1",
                    "score": 0.9,
                    "label": "RISK_DETECTED",
                    "detection_type": "chat"
                }
            ]
        }
        
        with patch.object(chat_detector, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response_data
            
            messages = [UserMessage(content="Test chat message", role="user")]
            result = await chat_detector.run_shield(shield_id=chat_detector.config.detector_id, messages=messages)
            
            assert isinstance(result, RunShieldResponse)
            mock_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_detect_chat_direct_mode(self):
        config = ChatDetectorConfig(
            detector_id="direct_chat",
            detector_url="https://api.example.com/chat",
            confidence_threshold=0.8
        )
        detector = ChatDetector(config)
        
        mock_response_data = [
            {
                "detections": [
                    {
                        "score": 0.9,
                        "label": "RISK_DETECTED",
                        "detection_type": "chat"
                    }
                ]
            }
        ]
        
        # Mock the shield store and shield
        mock_shield = MagicMock()
        mock_shield.identifier = detector.config.detector_id
        mock_shield_store = AsyncMock()
        mock_shield_store.get_shield.return_value = mock_shield
        detector.shield_store = mock_shield_store
        
        with patch.object(detector, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response_data
            
            messages = [UserMessage(content="Direct chat test", role="user")]
            result = await detector.run_shield(shield_id=detector.config.detector_id, messages=messages)
            
            assert isinstance(result, RunShieldResponse)

    def test_prepare_chat_request_orchestrator(self, chat_detector):
        messages = [
            {"content": "Hello", "role": "user"},
            {"content": "Hi!", "role": "system"}
        ]
        
        request_data = chat_detector._prepare_chat_request(messages)
        
        assert "messages" in request_data
        assert "detectors" in request_data
        assert len(request_data["messages"]) == 2
        
        # Check that nested detectors were flattened
        detectors = request_data["detectors"]
        raw_detectors = detectors['test_chat']['_raw_detectors']
        assert "detector_1" in raw_detectors
        assert raw_detectors["detector_1"]["detector_params"]["param1"] == "value1"

    def test_prepare_chat_request_direct(self):
        config = ChatDetectorConfig(
            detector_id="direct_chat",
            detector_url="https://api.example.com/chat"
        )
        detector = ChatDetector(config)
        
        messages = [{"content": "Direct test", "role": "user"}]
        request_data = detector._prepare_chat_request(messages)
        
        assert "messages" in request_data
        assert "detector_params" in request_data
        assert len(request_data["messages"]) == 1



class TestDetectorHTTP:
    @pytest.fixture
    def content_detector_with_httpx(self):
        config = ContentDetectorConfig(
            detector_id="http_test",
            detector_url="https://api.example.com/content",
            confidence_threshold=0.7
        )
        return ContentDetector(config)
    
    @pytest_asyncio.fixture
    async def initialized_detector(self, content_detector_with_httpx):
        detector = content_detector_with_httpx
        
        await detector.initialize()
        
        # Mock shield store
        mock_shield = MagicMock()
        mock_shield.identifier = detector.config.detector_id
        mock_shield_store = AsyncMock()
        mock_shield_store.get_shield.return_value = mock_shield
        detector.shield_store = mock_shield_store
        
        yield detector
        
        await detector.shutdown()

    @pytest.mark.asyncio
    async def test_http_request_success(self, initialized_detector, httpx_mock):
        detector = initialized_detector
        
        # Mock the HTTP response (API returns a list with detections)
        httpx_mock.add_response(
            method="POST",
            url="https://api.example.com/content/api/v1/text/contents",
            json=[{
                "score": 0.8,
                "label": "label",
                "detection_type": "content"
            }],
            status_code=200
        )
        
        messages = [UserMessage(content="Test content", role="user")]
        result = await detector.run_shield(shield_id=detector.config.detector_id, messages=messages)
        
        assert isinstance(result, RunShieldResponse)
        assert result.violation is not None  # Should detect violation because score > threshold

    @pytest.mark.asyncio
    async def test_http_request_no_violation(self, initialized_detector, httpx_mock):
        detector = initialized_detector
        
        httpx_mock.add_response(
            method="POST",
            url="https://api.example.com/content/api/v1/text/contents",
            json=[{
                "score": 0.3,
                "label": "label",
                "detection_type": "content"
            }],
            status_code=200
        )
        
        messages = [UserMessage(content="Safe content", role="user")]
        result = await detector.run_shield(shield_id=detector.config.detector_id, messages=messages)
        
        assert isinstance(result, RunShieldResponse)
        assert result.violation is None  # Should not detect violation because score < threshold

    @pytest.mark.asyncio
    async def test_http_request_error_handling(self, initialized_detector, httpx_mock):
        detector = initialized_detector
        
        # Mock a 500 server error for multiple retries (detector retries 3 times)
        for _ in range(3):
            httpx_mock.add_response(
                method="POST",
                url="https://api.example.com/content/api/v1/text/contents",
                json={"error": "Internal server error"},
                status_code=500
            )
        
        messages = [UserMessage(content="Test content", role="user")]
        
        # The error should be caught and wrapped in a RunShieldResponse with violation
        result = await detector.run_shield(shield_id=detector.config.detector_id, messages=messages)
        assert isinstance(result, RunShieldResponse)
        assert result.violation is not None
        assert "error" in result.violation.user_message.lower()

    @pytest.mark.asyncio
    async def test_empty_response_handling(self, initialized_detector, httpx_mock):
        detector = initialized_detector
        
        httpx_mock.add_response(
            method="POST",
            url="https://api.example.com/content/api/v1/text/contents",
            json=[],
            status_code=200
        )
        
        messages = [UserMessage(content="Test content", role="user")]
        result = await detector.run_shield(shield_id=detector.config.detector_id, messages=messages)
        
        assert isinstance(result, RunShieldResponse)
        assert result.violation is None


class TestDetectionResult:
    def test_detection_result_creation(self):
        result = DetectionResult(
            detection="detection",
            detection_type="content",
            score=0.8,
            detector_id="test_detector",
            metadata = {"risk_name": "test_risk", "category": "harmful"}
        )
        
        assert result.score == 0.8
        assert result.detection == "detection"
        assert result.detection_type == "content"
        assert result.detector_id == "test_detector"
        assert result.metadata["risk_name"] == "test_risk"
        
