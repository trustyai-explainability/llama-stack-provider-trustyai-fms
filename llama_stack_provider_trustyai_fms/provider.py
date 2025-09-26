import logging

from llama_stack.providers.datatypes import (
    AdapterSpec,
    Api,
    ProviderSpec,
    remote_provider_spec,
)

# Add logging at the top
logger = logging.getLogger(__name__)


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
