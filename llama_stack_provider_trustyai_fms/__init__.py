import logging
from typing import Any, Dict, Optional, Union

# First import the provider spec to ensure registration
from .provider import get_provider_spec

# Set up logging
logger = logging.getLogger(__name__)

# Import Safety API
from llama_stack.apis.safety import Safety
from llama_stack.providers.datatypes import Api

# Defer class imports by using a try-except block
try:
    from .config import (
        ChatDetectorConfig,
        ContentDetectorConfig,
        DetectorParams,
        EndpointType,
        FMSSafetyProviderConfig,
    )
    from .detectors.base import (
        BaseDetector,
        DetectorProvider,
    )
    from .detectors.chat import ChatDetector
    from .detectors.content import (
        ContentDetector,
    )

    # Type aliases for better readability
    ConfigType = Union[
        ContentDetectorConfig, ChatDetectorConfig, FMSSafetyProviderConfig
    ]
    DetectorType = Union[BaseDetector, DetectorProvider]
except ImportError:
    # These will be imported later when actually needed
    pass


class DetectorConfigError(ValueError):
    """Raised when detector configuration is invalid"""

    pass


async def create_fms_provider(config: Dict[str, Any]) -> Safety:
    """Create FMS safety provider instance.

    Args:
        config: Configuration dictionary

    Returns:
        Safety: Configured FMS safety provider
    """
    # Import here to avoid circular imports if needed
    from .config import FMSSafetyProviderConfig

    logger.debug("Creating trustyai-fms provider")
    return await get_adapter_impl(FMSSafetyProviderConfig(**config))


async def get_adapter_impl(
    config: FMSSafetyProviderConfig,
    deps: Dict[Api, Any],
) -> DetectorProvider:
    """Get appropriate detector implementation(s) based on config type."""
    try:
        detectors: Dict[str, Any] = {}

        # Process shields configuration
        for shield_id, shield_config in config.shields.items():
            impl: BaseDetector
            if isinstance(shield_config, ChatDetectorConfig):
                impl = ChatDetector(shield_config)
            elif isinstance(shield_config, ContentDetectorConfig):
                impl = ContentDetector(shield_config)
            else:
                raise DetectorConfigError(
                    f"Invalid shield config type for {shield_id}: {type(shield_config)}"
                )
            await impl.initialize()
            detectors[shield_id] = impl

        detectors_for_provider: Dict[str, BaseDetector] = {}
        for shield_id, detector in detectors.items():
            if isinstance(detector, BaseDetector):
                detectors_for_provider[shield_id] = detector

        return DetectorProvider(detectors_for_provider)

    except Exception as e:
        raise DetectorConfigError(
            f"Failed to create detector implementation: {str(e)}"
        ) from e


# Keep the __all__ list the same
__all__ = [
    # Factory methods
    "get_adapter_impl",
    "create_fms_provider",
    # Configurations
    "ContentDetectorConfig",
    "ChatDetectorConfig",
    "FMSSafetyProviderConfig",
    "EndpointType",
    "DetectorParams",
    # Implementations
    "ChatDetector",
    "ContentDetector",
    "BaseDetector",
    "DetectorProvider",
    # Types
    "ConfigType",
    "DetectorType",
    "DetectorConfigError",
]
