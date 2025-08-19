import logging

# Add logging at the top
logger = logging.getLogger(__name__)

from llama_stack.providers.datatypes import (
    ProviderSpec,
    Api,
    AdapterSpec,
    remote_provider_spec,
)


def get_provider_spec() -> ProviderSpec:
    return remote_provider_spec(
        api=Api.safety,
        adapter=AdapterSpec(
            adapter_type="trustyai_fms",
            config_class="llama_stack_provider_trustyai_fms.config.FMSSafetyProviderConfig",
            module="llama_stack_provider_trustyai_fms",
        ),
    )


def get_shields_provider_spec() -> ProviderSpec:
    spec = remote_provider_spec(
        api=Api.shields,
        adapter=AdapterSpec(
            adapter_type="trustyai_fms",
            config_class="llama_stack_provider_trustyai_fms.config.FMSSafetyProviderConfig",
            module="llama_stack_provider_trustyai_fms",
        ),
    )
    # Add debug logging
    logger.debug(f"Returning shields provider spec: {spec}")
    return spec