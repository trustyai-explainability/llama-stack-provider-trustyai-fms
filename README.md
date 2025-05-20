# `trustyai_fms`: out-of-tree remote safety provider for llama stack

This repo implements [FMS Guardrails Orchestrator](https://github.com/foundation-model-stack/fms-guardrails-orchestrator) together with community detectors: 

- [regex detectors](https://github.com/trustyai-explainability/guardrails-regex-detector)
- [Hugging Face content detectors](https://github.com/trustyai-explainability/guardrails-detectors)
- [vllm detector adapter](https://github.com/foundation-model-stack/vllm-detector-adapter)

as an out-of-tree remote safety provider for [llama stack](https://github.com/meta-llama/llama-stack) based on [this guideline](https://github.com/meta-llama/llama-stack/blob/main/docs/source/providers/external.md)

## Folder structure

```
.
├── llama_stack_provider_trustyai_fms
│   ├── __init__.py
│   ├── config.py
│   ├── detectors
│   │   ├── base.py
│   │   ├── chat.py
│   │   └── content.py
│   └── provider.py
├── notebooks
│   ├── trustyai-fms-detector-api.ipynb
│   └── trustyai-fms-orchestrator-api.ipynb
├── providers.d
│   └── remote
│       └── safety
│           └── trustyai_fms.yaml
├── pyproject.toml
├── README.md
└── runtime_configurations
    ├── detector_api.yaml
    └── orchestrator_api.yaml

```

- `llama_stack_provider_trustyai_fms/` -- the main package with the implementation of the remote safety provider
- `notebooks/` -- jupyter notebooks with examples of running shields once they are configured
- `providers.d` -- directory containing external provider specifications
- `runtime_configurations/` -- examples of  YAML file to configure the stack with the provider either using the [orchestrator API](https://foundation-model-stack.github.io/fms-guardrails-orchestrator/?urls.primaryName=Orchestrator+API) or the [detector API](https://foundation-model-stack.github.io/fms-guardrails-orchestrator/?urls.primaryName=Detector+API)

## Running demos

To run the demos in full, there is a need to deploy the orchestrator and detectors on Openshift, unless you have access to the necessary routes of the deployed services. If you do not have access to these routes, follow 
[Part A below](#part-a-openshift-setup-for-the-orchestrator-and-detectors) to set them up. 

Subsequently, to create a local llama stack distribution, follow [Part B below](#part-b-setup-to-create-a-local-llama-stack-distribution-with-external-trustyai_fms-remote-safety-provider)

### Part A. Openshift setup for the orchestrator and detectors

The demos require deploying the orchestrator and detectors on Openshift. 

The following operators are required in the Openshift cluster: 

__GPU__ -- follow [this guide](https://docs.nvidia.com/datacenter/cloud-native/openshift/latest/steps-overview.html) and install:
- Node Feature Discovery Operator (4.17.0-202505061137 provided by Red Hat):
    - ensure to create an instance of NodeFeatureDiscovery using the NodeFeatureDiscovery tab
- NVIDIA GPU Operator (25.3.0 provided by NVIDIA Corporation)
    - ensure to create an instance of ClusterPolicy using the ClusterPolicy tab

__Model Serving__: 
- Red Hat OpenShift Service Mesh 2 (2.6.7-0 provided by Red Hat, Inc.)
- Red Hat OpenShift Serverless (1.35.1 provided by Red Hat)
__Authentication__: 
- Red Hat - Authorino Operator (1.2.1 provided by Red Hat)

__AI Platform__:
- Red Hat OpenShift AI (2.20.0 provided by Red Hat, Inc.):
    - in the `DataScienceInitialization` resource, set the value of `managementState` for the `serviceMesh` component to `Removed`
    - in the `default-dsc`, ensure:
        1. `trustyai` `managementState` is set to `Managed`
        2. `kserve` is set to:
            ```yaml
            kserve:
                defaultDeploymentMode: RawDeployment
                managementState: Managed
                serving:
                    managementState: Removed
                    name: knative-serving
            ```

Once the above steps are completed, 

1. Create a new project
```bash
oc new-project test
```

2. Apply the manifests in the `openshift-manifests/` directory to deploy the orchestrator and detectors. 

```bash
oc apply -k openshift-manifests/
```

### Part B. Setup to create a local llama stack distribution with external trustyai_fms remote safety provider

1. Clone the repo
```bash
git clone https://github.com/m-misiura/llama-stack-provider-trustyai-fms.git
```

2. Change directory to the cloned repo
```bash
cd llama-stack-provider-trustyai-fms
```

3. Create a virtual environment
```bash
python3 -m venv .venv
```
4. Activate the virtual environment
```bash
source .venv/bin/activate
```
5. Install the requirements
```bash
pip install -e .
```

6. Pick a runtime configuration file from `runtime_configurations/` and run the stack: 

    a. __for the orchestrator API__:

    ```bash
    llama stack run runtime_configurations/orchestrator_api.yaml --image-type=venv
    ```

    Note that you might need to export the following environment variables: 

    ```bash
    export FMS_ORCHESTRATOR_URL="https://$(oc get routes guardrails-orchestrator-http -o jsonpath='{.spec.host}')"
    ```

    b. __for the detector API__:

    ```bash
    llama stack run runtime_configurations/detector_api.yaml --image-type=venv
    ```

    Not that you might need to export the following environment variables: 

    ```bash
    export FMS_CHAT_URL="http://$(oc get routes granite-2b-detector-route -o jsonpath='{.spec.host}')"
    export FMS_REGEX_URL="http://$(oc get routes pii-detector-route   -o jsonpath='{.spec.host}')"
    export FMS_HAP_URL="http://$(oc get routes hap-detector-route  -o jsonpath='{.spec.host}')"
    ```

7. Go through the notebook to see how to use the stack, e.g. in an other terminal open:
    - for __for the orchestrator API__:
    ```bash
    jupyter notebook notebooks/trustyai-fms-orchestrator-api.ipynb
    ```
    - for __for the detector API__:
    ```bash
    jupyter notebook noteboooks/trustyai-fms-detector-api.ipynb
    ```

