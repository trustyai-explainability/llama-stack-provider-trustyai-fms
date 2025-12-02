import logging

from .compat import Api, ProviderSpec, RemoteProviderSpec

logger = logging.getLogger(__name__)

# Check if we're using very old format (0.2.20-0.2.22 with AdapterSpec)
try:
    from llama_stack.providers.datatypes import AdapterSpec

    USE_ADAPTER_SPEC = True
except ImportError:
    # 0.3.3+ and llama-stack-api don't have AdapterSpec
    AdapterSpec = None
    USE_ADAPTER_SPEC = False


def get_provider_spec() -> ProviderSpec:
    """Get provider specification for Safety API.

    Compatible with llama-stack >= 0.2.20 and llama-stack-api >= 0.1.0.
    """
    if USE_ADAPTER_SPEC:
        # Very old (0.2.20-0.2.22): uses adapter field with AdapterSpec
        return RemoteProviderSpec(
            api=Api.safety,
            adapter=AdapterSpec(
                adapter_type="trustyai_fms",
                config_class="llama_stack_provider_trustyai_fms.config.FMSSafetyProviderConfig",
                module="llama_stack_provider_trustyai_fms",
            ),
            config_class="llama_stack_provider_trustyai_fms.config.FMSSafetyProviderConfig",
            provider_type="remote::trustyai_fms",
        )
    else:
        # Newer (0.3.3+) and new llama-stack-api: uses adapter_type field
        return RemoteProviderSpec(
            api=Api.safety,
            provider_type="remote::trustyai_fms",
            config_class="llama_stack_provider_trustyai_fms.config.FMSSafetyProviderConfig",
            module="llama_stack_provider_trustyai_fms",
            adapter_type="trustyai_fms",
        )


def get_shields_provider_spec() -> ProviderSpec:
    """Get provider specification for Shields API.

    Compatible with llama-stack >= 0.2.20 and llama-stack-api >= 0.1.0.
    """
    if USE_ADAPTER_SPEC:
        # Very old (0.2.20-0.2.22): uses adapter field with AdapterSpec
        return RemoteProviderSpec(
            api=Api.shields,
            adapter=AdapterSpec(
                adapter_type="trustyai_fms",
                config_class="llama_stack_provider_trustyai_fms.config.FMSSafetyProviderConfig",
                module="llama_stack_provider_trustyai_fms",
            ),
            config_class="llama_stack_provider_trustyai_fms.config.FMSSafetyProviderConfig",
            provider_type="remote::trustyai_fms",
        )
    else:
        # Newer (0.3.3+) and new llama-stack-api: uses adapter_type field
        return RemoteProviderSpec(
            api=Api.shields,
            provider_type="remote::trustyai_fms",
            config_class="llama_stack_provider_trustyai_fms.config.FMSSafetyProviderConfig",
            module="llama_stack_provider_trustyai_fms",
            adapter_type="trustyai_fms",
        )
