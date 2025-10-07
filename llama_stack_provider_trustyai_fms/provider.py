import logging

from llama_stack.providers.datatypes import Api, ProviderSpec

logger = logging.getLogger(__name__)

try:
    from llama_stack.providers.datatypes import AdapterSpec, remote_provider_spec

    USE_LEGACY = True
    logger.debug("Using legacy remote_provider_spec")
except ImportError:
    from llama_stack.providers.datatypes import RemoteProviderSpec

    USE_LEGACY = False
    logger.debug("Using new RemoteProviderSpec")


def get_provider_spec() -> ProviderSpec:
    if USE_LEGACY:
        return remote_provider_spec(
            api=Api.safety,
            adapter=AdapterSpec(
                adapter_type="trustyai_fms",
                config_class="llama_stack_provider_trustyai_fms.config.FMSSafetyProviderConfig",
                module="llama_stack_provider_trustyai_fms",
            ),
        )
    else:
        return RemoteProviderSpec(
            api=Api.safety,
            provider_type="remote::trustyai_fms",
            config_class="llama_stack_provider_trustyai_fms.config.FMSSafetyProviderConfig",
            module="llama_stack_provider_trustyai_fms",
            adapter_type="trustyai_fms",
        )


def get_shields_provider_spec() -> ProviderSpec:
    if USE_LEGACY:
        return remote_provider_spec(
            api=Api.shields,
            adapter=AdapterSpec(
                adapter_type="trustyai_fms",
                config_class="llama_stack_provider_trustyai_fms.config.FMSSafetyProviderConfig",
                module="llama_stack_provider_trustyai_fms",
            ),
        )
    else:
        return RemoteProviderSpec(
            api=Api.shields,
            provider_type="remote::trustyai_fms",
            config_class="llama_stack_provider_trustyai_fms.config.FMSSafetyProviderConfig",
            module="llama_stack_provider_trustyai_fms",
            adapter_type="trustyai_fms",
        )
