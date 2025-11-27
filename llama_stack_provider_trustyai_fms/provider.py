import logging

from llama_stack_api.datatypes import Api, ProviderSpec, RemoteProviderSpec

logger = logging.getLogger(__name__)


def get_provider_spec() -> ProviderSpec:
    """Get provider specification for Safety API.

    Returns RemoteProviderSpec for llama-stack-api >= 0.1.0.
    This provider requires llama-stack-api and is not compatible with legacy llama-stack.
    """
    return RemoteProviderSpec(
        api=Api.safety,
        provider_type="remote::trustyai_fms",
        config_class="llama_stack_provider_trustyai_fms.config.FMSSafetyProviderConfig",
        module="llama_stack_provider_trustyai_fms",
        adapter_type="trustyai_fms",
    )


def get_shields_provider_spec() -> ProviderSpec:
    """Get provider specification for Shields API.

    Returns RemoteProviderSpec for llama-stack-api >= 0.1.0.
    This provider requires llama-stack-api and is not compatible with legacy llama-stack.
    """
    return RemoteProviderSpec(
        api=Api.shields,
        provider_type="remote::trustyai_fms",
        config_class="llama_stack_provider_trustyai_fms.config.FMSSafetyProviderConfig",
        module="llama_stack_provider_trustyai_fms",
        adapter_type="trustyai_fms",
    )
