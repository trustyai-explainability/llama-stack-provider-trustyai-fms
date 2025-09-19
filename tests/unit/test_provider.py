from llama_stack.providers.datatypes import ProviderSpec, Api, AdapterSpec
from llama_stack_provider_trustyai_fms.provider import (
    get_provider_spec,
    get_shields_provider_spec
)


class TestProviderFunctions:
    def test_get_provider_spec(self):
        spec = get_provider_spec()
        
        assert isinstance(spec, ProviderSpec)
        assert spec.api == Api.safety
        assert isinstance(spec.adapter, AdapterSpec)
        assert spec.adapter.adapter_type == "trustyai_fms"
        assert spec.adapter.config_class == "llama_stack_provider_trustyai_fms.config.FMSSafetyProviderConfig"
        assert spec.adapter.module == "llama_stack_provider_trustyai_fms"

    def test_get_shields_provider_spec(self):
        spec = get_shields_provider_spec()

        assert isinstance(spec, ProviderSpec)
        assert spec.api == Api.shields
        assert isinstance(spec.adapter, AdapterSpec)
        assert spec.adapter.adapter_type == "trustyai_fms"
        assert spec.adapter.config_class == "llama_stack_provider_trustyai_fms.config.FMSSafetyProviderConfig"
        assert spec.adapter.module == "llama_stack_provider_trustyai_fms"
