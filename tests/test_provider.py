from llama_stack.providers.datatypes import Api, ProviderSpec

try:
    from llama_stack.providers.datatypes import AdapterSpec

    USE_LEGACY = True
except ImportError:
    from llama_stack.providers.datatypes import RemoteProviderSpec

    USE_LEGACY = False

from llama_stack_provider_trustyai_fms.provider import (
    get_provider_spec,
    get_shields_provider_spec,
)


class TestProviderFunctions:
    def test_get_provider_spec(self):
        spec = get_provider_spec()
        assert isinstance(spec, ProviderSpec)
        assert spec.api == Api.safety

        if USE_LEGACY:
            assert isinstance(spec.adapter, AdapterSpec)
            assert spec.adapter.adapter_type == "trustyai_fms"
            assert (
                spec.adapter.config_class
                == "llama_stack_provider_trustyai_fms.config.FMSSafetyProviderConfig"
            )
            assert spec.adapter.module == "llama_stack_provider_trustyai_fms"
        else:
            assert isinstance(spec, RemoteProviderSpec)
            assert spec.provider_type == "remote::trustyai_fms"
            assert spec.adapter_type == "trustyai_fms"
            assert (
                spec.config_class
                == "llama_stack_provider_trustyai_fms.config.FMSSafetyProviderConfig"
            )
            assert spec.module == "llama_stack_provider_trustyai_fms"

    def test_get_shields_provider_spec(self):
        spec = get_shields_provider_spec()
        assert isinstance(spec, ProviderSpec)
        assert spec.api == Api.shields

        if USE_LEGACY:
            assert isinstance(spec.adapter, AdapterSpec)
            assert spec.adapter.adapter_type == "trustyai_fms"
            assert (
                spec.adapter.config_class
                == "llama_stack_provider_trustyai_fms.config.FMSSafetyProviderConfig"
            )
            assert spec.adapter.module == "llama_stack_provider_trustyai_fms"
        else:
            assert isinstance(spec, RemoteProviderSpec)
            assert spec.provider_type == "remote::trustyai_fms"
            assert spec.adapter_type == "trustyai_fms"
            assert (
                spec.config_class
                == "llama_stack_provider_trustyai_fms.config.FMSSafetyProviderConfig"
            )
            assert spec.module == "llama_stack_provider_trustyai_fms"
