from __future__ import annotations

import asyncio
import datetime
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, ClassVar, Dict, List, Optional, Tuple, cast
from urllib.parse import urlparse

import httpx

from llama_stack.apis.inference import (
    CompletionMessage,
    Message,
    SystemMessage,
    ToolResponseMessage,
    UserMessage,
)
from llama_stack.apis.resource import ResourceType
from llama_stack.apis.safety import (
    RunShieldResponse,
    Safety,
    SafetyViolation,
    ShieldStore,
    ViolationLevel,
)
from llama_stack.apis.shields import ListShieldsResponse, Shield, Shields
from llama_stack.providers.datatypes import ShieldsProtocolPrivate
from ..config import (
    BaseDetectorConfig,
    EndpointType,
)

# Configure logging
logger = logging.getLogger(__name__)


# Custom exceptions
class DetectorError(Exception):
    """Base exception for detector errors"""

    pass


class DetectorConfigError(DetectorError):
    """Configuration related errors"""

    pass


class DetectorRequestError(DetectorError):
    """HTTP request related errors"""

    pass


class DetectorValidationError(DetectorError):
    """Validation related errors"""

    pass


class DetectorNetworkError(DetectorError):
    """Network connectivity issues"""


class DetectorTimeoutError(DetectorError):
    """Request timeout errors"""


class DetectorRateLimitError(DetectorError):
    """Rate limiting errors"""


class DetectorAuthError(DetectorError):
    """Authentication errors"""


# Type aliases
MessageDict = Dict[str, Any]
DetectorResponse = Dict[str, Any]
Headers = Dict[str, str]
RequestPayload = Dict[str, Any]


class MessageTypes(Enum):
    """Message type constants"""

    USER = auto()
    SYSTEM = auto()
    TOOL = auto()
    COMPLETION = auto()

    @classmethod
    def to_str(cls, value: MessageTypes) -> str:
        """Convert enum to string representation"""
        return value.name.lower()


@dataclass(frozen=True)
class DetectionResult:
    """Structured detection result"""

    detection: str
    detection_type: str
    score: float
    detector_id: str
    text: str = ""
    start: int = 0
    end: int = 0
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "detection": self.detection,
            "detection_type": self.detection_type,
            "score": self.score,
            "detector_id": self.detector_id,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            **({"metadata": self.metadata} if self.metadata else {}),
        }


class BaseDetector(Safety, ShieldsProtocolPrivate, ABC):
    """Base class for all safety detectors"""

    # Class constants
    VALID_SCHEMES: ClassVar[set] = {"http", "https"}

    def __init__(self, config: BaseDetectorConfig) -> None:
        """Initialize detector with configuration"""
        self.config = config
        self.registered_shields: List[Shield] = []
        self.score_threshold: float = config.confidence_threshold
        self._http_client: Optional[httpx.AsyncClient] = None
        self._shield_store: ShieldStore = SimpleShieldStore()
        self._validate_config()

    @property
    def shield_store(self) -> ShieldStore:
        """Get shield store instance"""
        if self._shield_store is None:
            self._shield_store = SimpleShieldStore()
        return self._shield_store

    @shield_store.setter
    def shield_store(self, value: ShieldStore) -> None:
        """Set shield store instance"""
        self._shield_store = value

    def _validate_config(self) -> None:
        """Validate detector configuration"""
        if not self.config:
            raise DetectorConfigError("Configuration is required")
        if not isinstance(self.config, BaseDetectorConfig):
            raise DetectorConfigError(f"Invalid config type: {type(self.config)}")

    async def initialize(self) -> None:
        """Initialize detector resources"""
        logger.info(f"Initializing {self.__class__.__name__}")
        self._http_client = httpx.AsyncClient(
            timeout=self.config.request_timeout,
            limits=httpx.Limits(
                max_keepalive_connections=self.config.max_keepalive_connections,
                max_connections=self.config.max_connections,
            ),
        )

    async def shutdown(self) -> None:
        """Clean up detector resources"""
        logger.info(f"Shutting down {self.__class__.__name__}")
        if self._http_client:
            await self._http_client.aclose()

    async def register_shield(self, shield: Shield) -> None:
        """Register a shield with the detector"""
        if not shield or not shield.identifier:
            raise DetectorValidationError("Invalid shield configuration")
        logger.info(f"Registering shield {shield.identifier}")
        self.registered_shields.append(shield)

    def _should_process_message(self, message: Message) -> bool:
        """Check if this detector should process the given message type"""
        # Get exact message type
        if isinstance(message, UserMessage):
            message_type = "user"
        elif isinstance(message, SystemMessage):
            message_type = "system"
        elif isinstance(message, ToolResponseMessage):
            message_type = "tool"
        elif isinstance(message, CompletionMessage):
            message_type = "completion"
        else:
            logger.warning(f"Unknown message type: {type(message)}")
            return False

        # Debug logging
        logger.debug(
            f"Message type check - type:'{message_type}', "
            f"config_types:{self.config.message_types}, "
            f"detector:{self.config.detector_id}"
        )

        # Explicit type check
        is_supported = message_type in self.config.message_types
        if not is_supported:
            logger.warning(
                f"Message type '{message_type}' not in configured types "
                f"{self.config.message_types} for detector {self.config.detector_id}"
            )
        return is_supported

    def _filter_messages(self, messages: List[Message]) -> List[Message]:
        """Filter messages based on configured message types"""
        return [msg for msg in messages if self._should_process_message(msg)]

    def _validate_url(self, url: str) -> None:
        """Validate URL format"""
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            raise DetectorConfigError(f"Invalid URL format: {url}")
        if parsed.scheme not in self.VALID_SCHEMES:
            raise DetectorConfigError(f"Invalid URL scheme: {parsed.scheme}")

    def _construct_url(self) -> str:
        """Construct API URL based on configuration"""
        if self.config.use_orchestrator_api:
            if not self.config.orchestrator_url:
                raise DetectorConfigError(
                    "orchestrator_url is required when use_orchestrator_api is True"
                )
            base_url = self.config.orchestrator_url
            endpoint_info = (
                EndpointType.ORCHESTRATOR_CHAT.value
                if self.config.is_chat
                else EndpointType.ORCHESTRATOR_CONTENT.value
            )
        else:
            if not self.config.detector_url:
                raise DetectorConfigError(
                    "detector_url is required when use_orchestrator_api is False"
                )
            base_url = self.config.detector_url
            endpoint_info = (
                EndpointType.DIRECT_CHAT.value
                if self.config.is_chat
                else EndpointType.DIRECT_CONTENT.value
            )

        url = f"{base_url.rstrip('/')}{endpoint_info['path']}"
        self._validate_url(url)
        logger.debug(
            f"Constructed URL: {url} for {'chat' if self.config.is_chat else 'content'} endpoint"
        )
        return url

    def _extract_detector_params(self) -> Dict[str, Any]:
        """Extract detector parameters from configuration"""
        detector_params: Dict[str, Any] = {}

        if (
            hasattr(self.config, "detector_params")
            and self.config.detector_params is not None
        ):
            # For chat detectors, extract model_params and metadata directly
            if hasattr(self.config.detector_params, "model_params"):
                detector_params.update(self.config.detector_params.model_params)

            if hasattr(self.config.detector_params, "metadata"):
                detector_params.update(self.config.detector_params.metadata)

            # Include any direct parameters
            for k, v in vars(self.config.detector_params).items():
                if v is not None and k not in [
                    "model_params",
                    "metadata",
                    "kwargs",
                    "params",
                ]:
                    if not (isinstance(v, (dict, list)) and len(v) == 0):
                        detector_params[k] = v

        return detector_params

    def _prepare_headers(self) -> Headers:
        """Prepare request headers based on configuration"""
        headers: Headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }

        if not self.config.use_orchestrator_api and self.config.detector_id:
            headers["detector-id"] = self.config.detector_id

        if self.config.auth_token:
            headers["Authorization"] = f"Bearer {self.config.auth_token}"

        return headers

    def _prepare_request_payload(
        self, messages: List[Message], params: Optional[Dict[str, Any]] = None
    ) -> RequestPayload:
        """Prepare request payload based on endpoint type and orchestrator mode"""
        logger.debug(
            f"Preparing payload - use_orchestrator: {self.config.use_orchestrator_api}, "
            f"detector_id: {self.config.detector_id}"
        )

        if self.config.use_orchestrator_api:
            payload: RequestPayload = {}

            # NEW STRUCTURE: Handle detectors at top level instead of under detector_params
            if hasattr(self.config, "detectors") and self.config.detectors:
                # Process the new structure with detectors at top level
                detector_config: Dict[str, Any] = {}
                for detector_id, det_config in self.config.detectors.items():
                    detector_config[detector_id] = det_config.get("detector_params", {})

                payload["detectors"] = detector_config

            # BACKWARD COMPATIBILITY: Handle legacy structures
            elif (
                hasattr(self.config, "detector_params")
                and self.config.detector_params is not None
            ):
                # Create detector configuration
                detector_config = {}

                # Extract parameters directly without wrapping them
                detector_params = {}

                # For chat detectors, extract model_params and metadata properly
                if hasattr(self.config.detector_params, "model_params"):
                    detector_params.update(self.config.detector_params.model_params)

                if hasattr(self.config.detector_params, "metadata"):
                    detector_params.update(self.config.detector_params.metadata)

                # Include direct parameters
                for k, v in vars(self.config.detector_params).items():
                    if v is not None and k not in [
                        "model_params",
                        "metadata",
                        "kwargs",
                        "params",
                        "detectors",
                    ]:
                        if not (isinstance(v, (dict, list)) and len(v) == 0):
                            detector_params[k] = v

                # Handle composite detectors
                if (
                    hasattr(self.config.detector_params, "detectors")
                    and self.config.detector_params.detectors
                ):
                    payload["detectors"] = self.config.detector_params.detectors
                else:
                    # Add detector configuration to payload
                    detector_config[self.config.detector_id] = detector_params
                    payload["detectors"] = detector_config

            # Add content or messages based on mode
            if self.config.is_chat:
                payload["messages"] = [msg.dict() for msg in messages]
            else:
                payload["content"] = messages[0].content

            logger.debug(f"Prepared orchestrator payload: {payload}")
            return payload
        else:
            # DIRECT MODE: Respect API-specific formats
            detector_params = self._extract_detector_params()

            # Extract parameters from nested containers if present
            flattened_params = {}

            # Handle complex parameter structures by flattening them for direct mode
            if isinstance(detector_params, dict):
                # First level: check for container structure
                for container_name in ["metadata", "model_params", "kwargs"]:
                    if container_name in detector_params:
                        # Extract and flatten parameters from containers
                        container = detector_params.get(container_name, {})
                        if isinstance(container, dict):
                            flattened_params.update(container)

                # If no container structure was found, use params directly
                if not flattened_params:
                    flattened_params = detector_params
            else:
                flattened_params = detector_params

            # Merge with any passed parameters
            if params:
                flattened_params.update(params)

            # Remove empty params dictionary if present
            if "params" in flattened_params and (
                flattened_params["params"] == {} or flattened_params["params"] is None
            ):
                del flattened_params["params"]

            if self.config.is_chat:
                payload = {
                    "messages": [msg.dict() for msg in messages],
                    "detector_params": flattened_params if flattened_params else {},
                }
            else:
                # For content APIs in direct mode, use plural form for compatibility
                payload = {
                    "contents": [
                        messages[0].content
                    ],  # Send as array for all content APIs
                    "detector_params": flattened_params if flattened_params else {},
                }

            logger.debug(f"Direct mode payload: {payload}")
            return payload

    async def _make_request(
        self,
        request: RequestPayload,
        headers: Optional[Headers] = None,
        timeout: Optional[float] = None,
    ) -> DetectorResponse:
        """Make HTTP request with error handling and retries"""
        if not self._http_client:
            raise DetectorError("HTTP client not initialized")

        url = self._construct_url()
        default_headers = self._prepare_headers()
        headers = {**default_headers, **(headers or {})}

        for attempt in range(self.config.max_retries):
            try:
                response = await self._http_client.post(
                    url,
                    json=request,
                    headers=headers,
                    timeout=timeout or self.config.request_timeout,
                )

                # Handle different error codes specifically
                if response.status_code == 429:
                    # Rate limit handling
                    retry_after = int(
                        response.headers.get(
                            "Retry-After", self.config.backoff_factor * 2
                        )
                    )
                    logger.warning(f"Rate limited, waiting {retry_after}s before retry")
                    await asyncio.sleep(retry_after)
                    continue

                elif response.status_code == 401:
                    raise DetectorAuthError(f"Authentication failed: {response.text}")

                elif response.status_code == 503:
                    # Service unavailable - return informative error if this is our last retry
                    if attempt == self.config.max_retries - 1:
                        error_details = {
                            "timestamp": datetime.datetime.now(
                                datetime.timezone.utc
                            ).isoformat(),
                            "service": urlparse(url).netloc,
                            "detector_id": self.config.detector_id,
                            "retries_attempted": self.config.max_retries,
                            "status_code": 503,
                        }

                        logger.error(
                            f"Service unavailable after {self.config.max_retries} attempts: "
                            f"{error_details['service']} for detector {self.config.detector_id}"
                        )

                        raise DetectorNetworkError(
                            f"Safety service is currently unavailable. The system attempted {self.config.max_retries}"
                            f"retries but couldn't connect to {error_details['service']}. Please try again "
                            f"later or contact your administrator if the problem persists."
                        )

                    # Continue with backoff if we have more retries
                    logger.warning(
                        f"Service unavailable (attempt {attempt+1}/{self.config.max_retries}), retrying..."
                    )
                else:
                    # SUCCESS PATH: Return immediately for successful responses
                    response.raise_for_status()
                    return cast(DetectorResponse, response.json())

            except httpx.TimeoutException as e:
                logger.error(
                    f"Request timed out (attempt {attempt + 1}/{self.config.max_retries})"
                )
                if attempt == self.config.max_retries - 1:
                    raise DetectorTimeoutError(
                        f"Request timed out after {self.config.max_retries} attempts"
                    ) from e

            except httpx.HTTPStatusError as e:
                # More specific error handling based on status code
                logger.error(
                    f"HTTP error {e.response.status_code} (attempt {attempt + 1}/{self.config.max_retries}): {e.response.text}"
                )
                if attempt == self.config.max_retries - 1:
                    raise DetectorRequestError(
                        f"API Error after {self.config.max_retries} attempts: {e.response.text}"
                    ) from e

            # Exponential backoff
            jitter = random.uniform(0.8, 1.2)
            await asyncio.sleep((self.config.backoff_factor**attempt) * jitter)
        raise DetectorRequestError(
            f"Request failed after {self.config.max_retries} attempts"
        )

    def _process_detection(
        self, detection: Dict[str, Any]
    ) -> Tuple[Optional[DetectionResult], float]:
        """Process detection result and return both result and score"""
        score = detection.get("score", 0.0)

        if "score" not in detection:
            logger.warning("Detection missing score field")
            return None, 0.0

        if score > self.score_threshold:
            return (
                DetectionResult(
                    detection="Yes",
                    detection_type=detection["detection_type"],
                    score=score,
                    detector_id=detection.get("detector_id", self.config.detector_id),
                    text=detection.get("text", ""),
                    start=detection.get("start", 0),
                    end=detection.get("end", 0),
                    metadata=detection.get("metadata"),
                ),
                score,
            )
        return None, score

    def create_violation_response(
        self,
        detection: DetectionResult,
        detector_id: str,
        level: ViolationLevel = ViolationLevel.ERROR,
    ) -> RunShieldResponse:
        """Create standardized violation response"""
        return RunShieldResponse(
            violation=SafetyViolation(
                user_message=f"Content flagged by {detector_id} as {detection.detection_type} with confidence {detection.score:.2f}",
                violation_level=level,
                metadata=detection.to_dict(),
            )
        )

    def _validate_shield(self, shield: Shield) -> None:
        """Validate shield configuration"""
        if not shield:
            raise DetectorValidationError("Shield not found")
        if not shield.identifier:
            raise DetectorValidationError("Shield missing identifier")

    @abstractmethod
    async def _run_shield_impl(
        self,
        shield_id: str,
        messages: List[Message],
        params: Optional[Dict[str, Any]] = None,
    ) -> RunShieldResponse:
        """Implementation specific shield running logic"""
        pass

    async def run_shield(
        self,
        shield_id: str,
        messages: List[Message],
        params: Optional[Dict[str, Any]] = None,
    ) -> RunShieldResponse:
        """Run safety checks using configured shield"""
        try:
            if not messages:
                return RunShieldResponse(
                    violation=SafetyViolation(
                        violation_level=ViolationLevel.INFO,
                        user_message="No messages to process",
                        metadata={"status": "skipped", "shield_id": shield_id},
                    )
                )

            supported_messages = []
            unsupported_types = set()

            for msg in messages:
                if self._should_process_message(msg):
                    supported_messages.append(msg)
                else:
                    msg_type = msg.type if hasattr(msg, "type") else type(msg).__name__
                    unsupported_types.add(msg_type)
                    logger.warning(
                        f"Message type '{msg_type}' not supported by shield {shield_id}. "
                        f"Allowed types: {list(self.config.message_types)}"
                    )

            if not supported_messages:
                return RunShieldResponse(
                    violation=SafetyViolation(
                        violation_level=ViolationLevel.WARN,
                        user_message=(
                            f"No supported message types to process. Shield {shield_id} only handles: "
                            f"{list(self.config.message_types)}"
                        ),
                        metadata={
                            "status": "skipped",
                            "error_type": "no_supported_messages",
                            "message_type": list(unsupported_types),
                            "supported_types": list(self.config.message_types),
                            "shield_id": shield_id,
                        },
                    )
                )

            # Step 4: Process supported messages
            return await self._run_shield_impl(shield_id, supported_messages, params)

        except Exception as e:
            logger.error(f"Shield execution failed: {str(e)}", exc_info=True)
            return RunShieldResponse(
                violation=SafetyViolation(
                    violation_level=ViolationLevel.ERROR,
                    user_message=f"Shield execution error: {str(e)}",
                    metadata={
                        "status": "error",
                        "error_type": "execution_error",
                        "shield_id": shield_id,
                        "error": str(e),
                    },
                )
            )


class SimpleShieldStore(ShieldStore):
    """Simplified shield store with caching"""

    def __init__(self):
        self._shields: Dict[str, Shield] = {}
        self._detector_configs = {}
        self._pending_configs = {}  # Add this to store configs before initialization
        self._store_id = id(self)
        self._initialized = False
        self._lock = asyncio.Lock()  # Add lock
        logger.info(f"Created SimpleShieldStore: {self._store_id}")

    async def register_detector_config(self, detector_id: str, config: Any) -> None:
        """Register detector configuration"""
        async with self._lock:
            if self._initialized:
                self._detector_configs[detector_id] = config
            else:
                self._pending_configs[detector_id] = config
            logger.info(
                f"Shield store {self._store_id} registered config for: {detector_id}"
            )

    async def update_shield_params(
        self, shield_id: str, params: Dict[str, Any]
    ) -> None:
        """Update shield parameters in the store"""
        shield = self._shields.get(shield_id)
        if shield:
            shield.params = params
            logger.debug(
                f"Shield store {self._store_id} updated params for shield {shield_id}: {params}"
            )

    # In the SimpleShieldStore class, replace the initialize method:

    async def initialize(self) -> None:
        """Initialize store and process pending configurations"""
        if self._initialized:
            return
        async with self._lock:
            # Process any pending configurations
            self._detector_configs.update(self._pending_configs)
            self._pending_configs.clear()
            self._initialized = True
            logger.info(
                f"Shield store {self._store_id} initialized with {len(self._detector_configs)} configs"
            )

        def _generate_params_for_shield(self, shield_id, config):
            """Helper method to generate shield parameters from config"""
            params = {}

            # For content detectors with multiple sub-detectors
            if hasattr(config, "detectors") and config.detectors:
                detectors_config = {}
                for det_id, det_config in config.detectors.items():
                    det_params = {}
                    if "detector_params" in det_config:
                        det_params = det_config["detector_params"]
                    detectors_config[det_id] = det_params
                params["detectors"] = detectors_config

            # For chat detectors with model parameters
            elif hasattr(config, "detector_params") and config.detector_params:
                if (
                    hasattr(config.detector_params, "model_params")
                    and config.detector_params.model_params
                ):
                    params["model_params"] = dict(config.detector_params.model_params)

                # Add metadata fields
                if (
                    hasattr(config.detector_params, "metadata")
                    and config.detector_params.metadata
                ):
                    if "model_params" not in params:
                        params["model_params"] = {}
                    # Add all metadata fields
                    for key, value in config.detector_params.metadata.items():
                        params["model_params"][key] = value

            # Add common shield information
            params.update(
                {
                    "display_name": f"{shield_id} Shield",
                    "display_description": f"Safety shield for {shield_id}",
                    "detector_type": "content" if not config.is_chat else "chat",
                    "message_types": list(config.message_types),
                    "confidence_threshold": config.confidence_threshold,
                }
            )

            return params

    async def get_shield(self, identifier: str) -> Shield:
        """Get or create shield by identifier"""
        await self.initialize()

        # Convert to string if needed
        identifier = str(identifier)

        if identifier in self._shields:
            logger.debug(
                f"Shield store {self._store_id} found existing shield: {identifier}"
            )
            return self._shields[identifier]

        config = self._detector_configs.get(identifier)
        if config:
            logger.info(
                f"Shield store {self._store_id} creating shield for {identifier} using config"
            )

            # Create shield params dictionary
            params = {}

            # For content detectors with multiple sub-detectors (email_hap)
            if hasattr(config, "detectors") and config.detectors:
                detectors_config = {}
                for det_id, det_config in config.detectors.items():
                    det_params = {}
                    if "detector_params" in det_config:
                        # Include all parameters from detector_params
                        det_params = det_config["detector_params"]
                    detectors_config[det_id] = det_params

                # Add detectors to params
                params["detectors"] = detectors_config

            # For chat detectors like granite with model parameters
            elif hasattr(config, "detector_params") and config.detector_params:
                if (
                    hasattr(config.detector_params, "model_params")
                    and config.detector_params.model_params
                ):
                    params["model_params"] = dict(config.detector_params.model_params)

                # Add metadata fields as part of model_params for chat detectors
                if (
                    hasattr(config.detector_params, "metadata")
                    and config.detector_params.metadata
                ):
                    if "model_params" not in params:
                        params["model_params"] = {}
                    # Add all metadata fields
                    for key, value in config.detector_params.metadata.items():
                        params["model_params"][key] = value

            # Add common shield information
            params.update(
                {
                    "display_name": f"{identifier} Shield",
                    "display_description": f"Safety shield for {identifier}",
                    "detector_type": "content" if not config.is_chat else "chat",
                    "message_types": list(config.message_types),
                    "confidence_threshold": config.confidence_threshold,
                }
            )

            # Create shield with proper params
            shield = Shield(
                identifier=identifier,
                provider_id="trustyai_fms",
                provider_resource_id=identifier,
                type=ResourceType.shield.value,
                params=params,
            )

            logger.info(
                f"Shield store {self._store_id} created shield: {identifier} with params: {params}"
            )
            self._shields[identifier] = shield
            return shield
        else:
            # Fail explicitly if no config found
            logger.error(
                f"Shield store {self._store_id} failed to create shield: no configuration found for {identifier}"
            )
            raise DetectorValidationError(
                f"Cannot create shield '{identifier}': no detector configuration found. "
                "Shields must have a valid detector configuration to ensure proper safety checks."
            )

    async def list_shields(self) -> ListShieldsResponse:
        """List all registered shields with their parameters"""
        if not self._initialized:
            # Ensure provider is initialized, which populates self._shields
            # with Shield instances that include their parameters.
            logger.info(  # Changed to INFO
                f"Provider {self._provider_id} - list_shields: Not initialized, calling initialize()."
            )
            await self.initialize()
        else:
            logger.info(
                f"Provider {self._provider_id} - list_shields: Already initialized."
            )

        # Get all shield instances directly from the _shields dictionary values.
        shields_to_return = list(self._shields.values())
        logger.info(
            f"Provider {self._provider_id} - list_shields: Retrieved {len(shields_to_return)} shields from self._shields.values()."
        )

        # Log what we're returning for debugging purposes
        shield_ids = [s.identifier for s in shields_to_return]
        logger.info(
            f"Provider {self._provider_id} listing {len(shields_to_return)} shields: {shield_ids}"
        )

        # Detailed debug log for each shield being returned, specifically checking params
        if not shields_to_return:
            logger.info(
                f"Provider {self._provider_id} - list_shields: No shields to return."
            )

        # Detailed debug log for each shield being returned, specifically checking params
        if not shields_to_return:
            logger.info(
                f"Provider {self._provider_id} - list_shields: No shields to return."
            )
        else:  # Added else to ensure the detailed check log title always appears if there are shields
            logger.info(
                f"Provider {self._provider_id} - list_shields - DETAILED PARAM CHECK BEFORE RESPONSE:"
            )
            for shield_debug_loop in shields_to_return:
                logger.info(
                    f"Provider {self._provider_id} - Shield ID='{shield_debug_loop.identifier}', Params='{shield_debug_loop.params}', Object ID='{id(shield_debug_loop)}'"
                )

        return ListShieldsResponse(data=shields_to_return)


class DetectorProvider(Safety, Shields):
    """Provider for managing safety detectors and shields"""

    def __init__(self, detectors: Dict[str, BaseDetector]) -> None:
        self.detectors = detectors
        self._shield_store: ShieldStore = SimpleShieldStore()
        self._shields: Dict[str, Shield] = {}
        self._initialized = False
        self._provider_id = id(self)
        self._detector_key_to_id = {}  # Add mapping dict
        self._pending_configs = []  # Store configurations for later registration

        # Store configurations for async registration
        for detector_key, detector in detectors.items():
            detector.shield_store = self._shield_store
            config_id = detector.config.detector_id
            self._detector_key_to_id[detector_key] = config_id
            self._pending_configs.append((config_id, detector.config))
        logger.info(
            f"Created DetectorProvider {self._provider_id} with {len(detectors)} detectors"
        )

    @property
    def shield_store(self) -> ShieldStore:
        return self._shield_store

    @shield_store.setter
    def shield_store(self, value: ShieldStore) -> None:
        """Set shield store instance"""
        if not value:
            logger.warning(f"Provider {self._provider_id} received null shield store")
            return

        logger.info(
            f"Provider {self._provider_id} setting new shield store: {id(value)}"
        )
        self._shield_store = value

        # Update detectors and sync shields
        for detector_id, detector in self.detectors.items():
            detector.shield_store = value
            logger.debug(
                f"Provider {self._provider_id} updated detector {detector_id} with shield store {id(value)}"
            )

            # Register detector configs if possible using getattr for safe access
            if hasattr(value, "register_detector_config") and hasattr(
                detector, "config"
            ):
                # Use getattr to get the method safely
                register_method = getattr(value, "register_detector_config", None)
                if callable(register_method):
                    asyncio.create_task(
                        register_method(detector.config.detector_id, detector.config)
                    )

    # Add this helper method to the DetectorProvider class
    def _prepare_shield_for_storage(self, shield: Shield) -> dict:
        """Prepare shield for storage by ensuring all attributes are properly serialized"""
        # Create a complete dictionary manually to ensure all fields are included
        shield_dict = {
            "identifier": shield.identifier,
            "provider_resource_id": shield.provider_resource_id,
            "provider_id": shield.provider_id,
            "type": shield.type,
            "access_attributes": shield.access_attributes,
        }

        # Always ensure params exist and are properly copied
        if shield.params:
            # Make a deep copy to avoid reference issues
            shield_dict["params"] = dict(shield.params)
        else:
            # Generate parameters if missing
            detector = next(
                (
                    d
                    for d in self.detectors.values()
                    if d.config.detector_id == shield.identifier
                ),
                None,
            )
            if detector:
                shield_dict["params"] = self._generate_shield_params(detector)
            else:
                shield_dict["params"] = {}

        return shield_dict

    async def initialize(self) -> None:
        """Initialize provider and register initial shields"""
        if self._initialized:
            return

        logger.info(f"Provider {self._provider_id} starting initialization")

        try:
            # First register all configurations with shield store
            if hasattr(self._shield_store, "register_detector_config"):
                tasks = []
                for config_id, config in self._pending_configs:
                    tasks.append(
                        self._shield_store.register_detector_config(config_id, config)
                    )

                if tasks:
                    await asyncio.gather(*tasks)

            # Initialize detectors
            for detector in self.detectors.values():
                await detector.initialize()

            # Create shields using the shield store
            for detector_id, detector in self.detectors.items():
                config_id = detector.config.detector_id
                shield = await self._shield_store.get_shield(config_id)

                # Log shield details to debug
                logger.info(f"Created shield {config_id} with params: {shield.params}")

                # Store the shield
                self._shields[config_id] = shield

                # Register shield with detector
                await detector.register_shield(shield)

            self._initialized = True
            logger.info(
                f"Provider {self._provider_id} initialization complete with {len(self._shields)} shields"
            )

        except Exception as e:
            logger.error(f"Provider {self._provider_id} initialization failed: {e}")
            raise

    # In DetectorProvider class

    async def list_shields(self) -> ListShieldsResponse:
        """List all registered shields with their parameters"""
        if not self._initialized:
            # Ensure provider is initialized, which populates self._shields
            # with Shield instances that include their parameters.
            logger.debug(
                f"Provider {self._provider_id} - list_shields: Not initialized, calling initialize()."
            )
            await self.initialize()
        else:
            logger.debug(
                f"Provider {self._provider_id} - list_shields: Already initialized."
            )

        # Get all shield instances directly from the _shields dictionary values.
        shields_to_return = list(self._shields.values())
        logger.debug(
            f"Provider {self._provider_id} - list_shields: Retrieved {len(shields_to_return)} shields from self._shields.values()."
        )

        # Log what we're returning for debugging purposes
        shield_ids = [s.identifier for s in shields_to_return]
        logger.info(
            f"Provider {self._provider_id} listing {len(shields_to_return)} shields: {shield_ids}"
        )

        # Detailed debug log for each shield being returned, specifically checking params
        if not shields_to_return:
            logger.debug(
                f"Provider {self._provider_id} - list_shields: No shields to return."
            )

        logger.debug(
            f"Provider {self._provider_id} - list_shields - DETAILED PARAM CHECK BEFORE RESPONSE:"
        )
        for shield_debug_loop in shields_to_return:
            logger.debug(
                f"Provider {self._provider_id} - Shield ID='{shield_debug_loop.identifier}', Params='{shield_debug_loop.params}', Object ID='{id(shield_debug_loop)}'"
            )

        return ListShieldsResponse(data=shields_to_return)

    async def get_shield(self, identifier: str) -> Shield:
        """Get shield by identifier"""
        await self.initialize()

        # Return existing shield
        if identifier in self._shields:
            return self._shields[identifier]

        # Get detector and config
        detector = self.detectors.get(identifier)
        if not detector:
            raise DetectorValidationError(f"Shield not found: {identifier}")

        # Create shield from store
        shield = await self._shield_store.get_shield(identifier)

        # Ensure shield has parameters
        if shield and not shield.params:
            # Generate parameters based on detector configuration
            if detector.config.is_chat:
                params = {
                    "display_name": f"{identifier} Shield",
                    "display_description": f"Safety shield for {identifier}",
                    "detector_type": "chat",
                    "message_types": list(detector.config.message_types),
                    "confidence_threshold": detector.config.confidence_threshold,
                }

                # Add model params if available
                if (
                    hasattr(detector.config, "detector_params")
                    and detector.config.detector_params
                ):
                    if hasattr(detector.config.detector_params, "model_params"):
                        params["model_params"] = dict(
                            detector.config.detector_params.model_params or {}
                        )

                    # Add metadata if available
                    if hasattr(detector.config.detector_params, "metadata"):
                        if "model_params" not in params:
                            params["model_params"] = {}
                        for (
                            key,
                            value,
                        ) in detector.config.detector_params.metadata.items():
                            params["model_params"][key] = value
            else:
                params = {
                    "display_name": f"{identifier} Shield",
                    "display_description": f"Safety shield for {identifier}",
                    "detector_type": "content",
                    "message_types": list(detector.config.message_types),
                    "confidence_threshold": detector.config.confidence_threshold,
                }

                # Add detector configuration for composite detectors
                if hasattr(detector.config, "detectors") and detector.config.detectors:
                    params["detectors"] = {
                        det_id: det_config.get("detector_params", {})
                        for det_id, det_config in detector.config.detectors.items()
                    }

            # Set the parameters on the shield
            shield.params = params
            logger.debug(
                f"Generated missing parameters for shield {identifier}: {params}"
            )

        if shield:
            self._shields[identifier] = shield
            return shield

        raise DetectorValidationError(f"Failed to get shield: {identifier}")

    # In DetectorProvider class

    async def register_shield(
        self,
        shield_id: str,
        provider_shield_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Shield:
        """Register a new shield"""
        if not self._initialized:
            await self.initialize()

        # Get the string identifier regardless of input type
        if hasattr(shield_id, "identifier"):
            shield_identifier = shield_id.identifier
        else:
            shield_identifier = str(shield_id)

        logger.debug(f"Registering shield with ID: {shield_identifier}")

        # Check if shield already exists by string identifier
        if shield_identifier in self._shields:
            shield = self._shields[shield_identifier]
            logger.debug(
                f"Shield {shield_identifier} already registered, returning existing instance with params: {shield.params}"
            )
            return shield

        # Get or create shield from store
        shield = await self._shield_store.get_shield(shield_identifier)
        if not shield:
            raise DetectorValidationError(
                f"Failed to create shield: {shield_identifier}"
            )

        # Update fields if provided
        if provider_id:
            shield.provider_id = provider_id
        if provider_shield_id:
            shield.provider_resource_id = provider_shield_id
        if params is not None:
            shield.params = params

        # Ensure shield parameters exist even if not provided
        if not shield.params:
            detector = next(
                (
                    d
                    for d in self.detectors.values()
                    if d.config.detector_id == shield_identifier
                ),
                None,
            )
            if detector:
                shield.params = self._generate_shield_params(detector)
                logger.debug(
                    f"Generated missing parameters for shield {shield_identifier}: {shield.params}"
                )

        # Store shield by string identifier
        self._shields[shield_identifier] = shield
        logger.debug(
            f"Shield {shield_identifier} registered with params: {shield.params}"
        )
        # Register with detectors
        for detector in self.detectors.values():
            await detector.register_shield(shield)

        return shield

    def _generate_shield_params(self, detector) -> Dict[str, Any]:
        """Generate shield parameters from detector config"""
        shield_id = detector.config.detector_id

        if detector.config.is_chat:
            # Chat detector params
            params = {
                "display_name": f"{shield_id} Shield",
                "display_description": f"Safety shield for {shield_id}",
                "detector_type": "chat",
                "message_types": list(detector.config.message_types),
                "confidence_threshold": detector.config.confidence_threshold,
            }

            # Add model params if available
            if (
                hasattr(detector.config, "detector_params")
                and detector.config.detector_params
            ):
                if hasattr(detector.config.detector_params, "model_params"):
                    params["model_params"] = dict(
                        detector.config.detector_params.model_params or {}
                    )

                # Add metadata if available
                if hasattr(detector.config.detector_params, "metadata"):
                    for key, value in detector.config.detector_params.metadata.items():
                        if "model_params" not in params:
                            params["model_params"] = {}
                        params["model_params"][key] = value
        else:
            # Content detector params
            params = {
                "display_name": f"{shield_id} Shield",
                "display_description": f"Safety shield for {shield_id}",
                "detector_type": "content",
                "message_types": list(detector.config.message_types),
                "confidence_threshold": detector.config.confidence_threshold,
            }

            # Add detector configuration for composite detectors
            if hasattr(detector.config, "detectors") and detector.config.detectors:
                params["detectors"] = {
                    det_id: det_config.get("detector_params", {})
                    for det_id, det_config in detector.config.detectors.items()
                }

        return params

    async def run_shield(
        self,
        shield_id: str,
        messages: List[Message],
        params: Optional[Dict[str, Any]] = None,
    ) -> RunShieldResponse:
        """Run shield against messages with enhanced composite handling"""
        try:
            # Step 1: Initial validation and initialization
            if not self._initialized:
                await self.initialize()

            if not messages:
                return RunShieldResponse(
                    violation=SafetyViolation(
                        violation_level=ViolationLevel.INFO,
                        user_message="No messages to process",
                        metadata={"status": "skipped", "shield_id": shield_id},
                    )
                )

            # Step 2: Get and validate shield configuration
            shield_detectors = [
                detector
                for detector in self.detectors.values()
                if detector.config.detector_id == shield_id
            ]

            if not shield_detectors:
                return RunShieldResponse(
                    violation=SafetyViolation(
                        violation_level=ViolationLevel.ERROR,
                        user_message=f"No detectors found for shield: {shield_id}",
                        metadata={
                            "status": "error",
                            "error_type": "detectors_not_found",
                            "shield_id": shield_id,
                        },
                    )
                )

            detector = shield_detectors[0]

            # Step 3: Filter messages and track skipped ones
            skipped_messages = []
            filtered_messages = []

            for idx, msg in enumerate(messages):
                if detector._should_process_message(msg):
                    filtered_messages.append((idx, msg))
                else:
                    msg_type = msg.type if hasattr(msg, "type") else type(msg).__name__
                    skipped_messages.append(
                        {
                            "index": idx,
                            "type": msg_type,
                            "reason": f"Message type '{msg_type}' not supported",
                        }
                    )

            if not filtered_messages:
                return RunShieldResponse(
                    violation=SafetyViolation(
                        violation_level=ViolationLevel.WARN,
                        user_message=(
                            f"No supported message types to process. Shield {shield_id} only handles: "
                            f"{list(detector.config.message_types)}"
                        ),
                        metadata={
                            "status": "skipped",
                            "error_type": "no_supported_messages",
                            "supported_types": list(detector.config.message_types),
                            "shield_id": shield_id,
                            "skipped_messages": skipped_messages,
                        },
                    )
                )

            # Step 4: Initialize result tracking
            message_results = []
            has_violation = False
            highest_violation_score = 0.0
            total_detections = 0

            # Step 5: Determine if detector is composite
            is_composite = (
                detector.config.detector_params is not None
                and hasattr(detector.config.detector_params, "detectors")
                and detector.config.detector_params.detectors is not None
            )

            # Step 6: Process messages in parallel
            if is_composite:
                # Define function to process composite detector message
                async def process_composite_message(orig_idx, message):
                    try:
                        current_result = {
                            "message_index": orig_idx,
                            "text": message.content,
                            "status": "pass",
                            "score": None,
                            "detection_type": None,
                        }

                        # Make API request for this message
                        request = detector._prepare_request_payload([message], params)
                        response = await detector._make_request(request)
                        detections = response.get("detections", [])
                        configured_detectors = []
                        if (
                            detector.config.detector_params is not None
                            and hasattr(detector.config.detector_params, "detectors")
                            and detector.config.detector_params.detectors is not None
                        ):
                            configured_detectors = list(
                                detector.config.detector_params.detectors.keys()
                            )

                        individual_results = []
                        message_has_violation = False
                        message_highest_score = 0.0
                        message_detections = 0

                        for det_id in configured_detectors:
                            detection = next(
                                (
                                    d
                                    for d in detections
                                    if d.get("detector_id") == det_id
                                ),
                                None,
                            )

                            score = detection.get("score") if detection else None
                            is_violation = (
                                score is not None and score > detector.score_threshold
                            )

                            if is_violation:
                                message_has_violation = True
                                message_detections += 1
                                if score > message_highest_score:
                                    message_highest_score = score
                                    current_result.update(
                                        {
                                            "status": "violation",
                                            "score": score,
                                            "detection_type": detection.get(
                                                "detection_type"
                                            ),
                                        }
                                    )

                            individual_results.append(
                                {
                                    "detector_id": det_id,
                                    "status": "violation" if is_violation else "pass",
                                    "score": score,
                                    "detection_type": (
                                        detection.get("detection_type")
                                        if detection
                                        else None
                                    ),
                                }
                            )

                        current_result["individual_detector_results"] = (
                            individual_results
                        )

                        return {
                            "result": current_result,
                            "has_violation": message_has_violation,
                            "highest_score": message_highest_score,
                            "detections": message_detections,
                        }
                    except Exception as e:
                        logger.error(
                            f"Message processing failed for message {orig_idx}: {e}"
                        )
                        return {
                            "result": {
                                "message_index": orig_idx,
                                "text": message.content if message else "",
                                "status": "error",
                                "error": str(e),
                            },
                            "has_violation": False,
                            "highest_score": 0.0,
                            "detections": 0,
                            "error": str(e),
                        }

                # Create tasks for all messages with controlled concurrency
                # Use semaphore to limit concurrent API calls
                semaphore = asyncio.Semaphore(detector.config.max_concurrency)

                async def process_with_semaphore(orig_idx, message):
                    async with semaphore:
                        return await process_composite_message(orig_idx, message)

                # Create and execute tasks
                tasks = [
                    process_with_semaphore(orig_idx, message)
                    for orig_idx, message in filtered_messages
                ]

                # Await all tasks
                task_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                for result in task_results:
                    if isinstance(result, Exception):
                        # Handle unexpected exceptions
                        logger.error(f"Task execution failed: {result}")
                        continue

                    # Extract data
                    assert isinstance(
                        result, dict
                    ), "Expected result to be a dictionary"
                    message_results.append(result["result"])

                    # Update aggregate metrics
                    if result["has_violation"]:
                        has_violation = True
                        total_detections += result["detections"]
                        if result["highest_score"] > highest_violation_score:
                            highest_violation_score = result["highest_score"]

            else:
                # For non-composite detectors
                async def process_standard_message(orig_idx, message):
                    try:
                        current_result = {
                            "message_index": orig_idx,
                            "text": message.content,
                            "status": "pass",
                            "score": None,
                            "detection_type": None,
                        }

                        # Make API request for this message
                        response = await detector._run_shield_impl(
                            shield_id, [message], params
                        )

                        if response.violation:
                            score = response.violation.metadata.get("score")
                            current_result.update(
                                {
                                    "status": "violation",
                                    "score": score,
                                    "detection_type": response.violation.metadata.get(
                                        "detection_type"
                                    ),
                                }
                            )

                            return {
                                "result": current_result,
                                "has_violation": True,
                                "highest_score": score or 0.0,
                                "detections": 1,
                            }

                        return {
                            "result": current_result,
                            "has_violation": False,
                            "highest_score": 0.0,
                            "detections": 0,
                        }
                    except Exception as e:
                        logger.error(
                            f"Message processing failed for message {orig_idx}: {e}"
                        )
                        return {
                            "result": {
                                "message_index": orig_idx,
                                "text": message.content if message else "",
                                "status": "error",
                                "error": str(e),
                            },
                            "has_violation": False,
                            "highest_score": 0.0,
                            "detections": 0,
                            "error": str(e),
                        }

                # Create tasks with controlled concurrency
                semaphore = asyncio.Semaphore(detector.config.max_concurrency)

                async def process_with_semaphore(orig_idx, message):
                    async with semaphore:
                        return await process_standard_message(orig_idx, message)

                # Create and execute tasks
                tasks = [
                    process_with_semaphore(orig_idx, message)
                    for orig_idx, message in filtered_messages
                ]

                # Await all tasks
                task_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                for result in task_results:
                    if isinstance(result, Exception):
                        # Handle unexpected exceptions
                        logger.error(f"Task execution failed: {result}")
                        continue

                    # Extract data
                    assert isinstance(
                        result, dict
                    ), "Expected result to be a dictionary"
                    message_results.append(result["result"])

                    # Update aggregate metrics
                    if result["has_violation"]:
                        has_violation = True
                        total_detections += result["detections"]
                        if result["highest_score"] > highest_violation_score:
                            highest_violation_score = result["highest_score"]

            # Step 7: Calculate summary statistics
            total_filtered = len(filtered_messages)
            violated_messages = sum(
                1 for r in message_results if r["status"] == "violation"
            )
            passed_messages = total_filtered - violated_messages

            message_pass_rate = round(
                passed_messages / total_filtered if total_filtered > 0 else 0,
                3,
            )
            message_fail_rate = round(
                violated_messages / total_filtered if total_filtered > 0 else 0,
                3,
            )

            # Step 8: Prepare summary
            summary = {
                "total_messages": len(messages),
                "processed_messages": total_filtered,
                "skipped_messages": len(skipped_messages),
                "messages_with_violations": violated_messages,
                "messages_passed": passed_messages,
                "message_fail_rate": message_fail_rate,
                "message_pass_rate": message_pass_rate,
                "total_detections": total_detections,
                "detector_breakdown": {
                    "active_detectors": (
                        len(detector.config.detector_params.detectors)
                        if is_composite
                        and detector.config.detector_params is not None
                        and hasattr(detector.config.detector_params, "detectors")
                        and detector.config.detector_params.detectors is not None
                        else 1
                    ),
                    "total_checks_performed": (
                        total_filtered * len(detector.config.detector_params.detectors)
                        if is_composite
                        and detector.config.detector_params is not None
                        and hasattr(detector.config.detector_params, "detectors")
                        and detector.config.detector_params.detectors is not None
                        else total_filtered
                    ),
                    "total_violations_found": total_detections,
                    "violations_per_message": round(
                        total_detections / total_filtered if total_filtered > 0 else 0,
                        3,
                    ),
                },
            }

            # Step 9: Prepare metadata
            metadata = {
                "status": "violation" if has_violation else "pass",
                "shield_id": shield_id,
                "confidence_threshold": detector.score_threshold,
                "summary": summary,
                "results": message_results,
            }

            # Step 10: Prepare response message
            skipped_msg = (
                f" ({len(skipped_messages)} messages skipped)"
                if skipped_messages
                else ""
            )
            base_msg = (
                f"Content violation detected by shield {shield_id} "
                f"(confidence: {highest_violation_score:.2f}, "
                f"{violated_messages}/{total_filtered} processed messages violated)"
                if has_violation
                else f"Content verified by shield {shield_id} "
                f"({total_filtered} messages processed)"
            )

            # Step 11: Return final response
            return RunShieldResponse(
                violation=SafetyViolation(
                    violation_level=(
                        ViolationLevel.ERROR if has_violation else ViolationLevel.INFO
                    ),
                    user_message=f"{base_msg}{skipped_msg}",
                    metadata=metadata,
                )
            )

        except Exception as e:
            logger.error(f"Shield execution failed: {str(e)}", exc_info=True)
            return RunShieldResponse(
                violation=SafetyViolation(
                    violation_level=ViolationLevel.ERROR,
                    user_message=f"Shield execution error: {str(e)}",
                    metadata={
                        "status": "error",
                        "error_type": "execution_error",
                        "shield_id": shield_id,
                        "error": str(e),
                    },
                )
            )

    async def shutdown(self) -> None:
        """Cleanup resources"""
        logger.info(f"Provider {self._provider_id} shutting down")
        errors = []

        for detector_id, detector in self.detectors.items():
            try:
                await detector.shutdown()
                logger.debug(
                    f"Provider {self._provider_id} shutdown detector: {detector_id}"
                )
            except Exception as e:
                error_msg = f"Error shutting down detector {detector_id}: {e}"
                logger.error(f"Provider {self._provider_id} {error_msg}")
                errors.append(error_msg)

        if errors:
            raise DetectorError(
                f"Provider {self._provider_id} shutdown errors: {', '.join(errors)}"
            )

        logger.info(f"Provider {self._provider_id} shutdown complete")
