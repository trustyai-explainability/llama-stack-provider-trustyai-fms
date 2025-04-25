## Steps to create a local llama stack distro with external trustyai_fms remote safety provider

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
python3 -m venv venv
```
4. Activate the virtual environment
```bash
source venv/bin/activate
```
5. Install the requirements
```bash
pip install -e .
```

6. Before running `orchestrator_api.yaml`

- make sure you have followed the instructions in [this repo](https://github.com/m-misiura/deployments-4-guardrails/tree/main) to deploy the necessary components

7. Set the `FMS_ORCHESTRATOR_URL` from your OpenShift cluster
```bash
export FMS_ORCHESTRATOR_URL="https://$(oc get routes guardrails-nlp -o jsonpath='{.spec.host}')"
```

8. After deploying the components, run the `orchestrator_api.yaml` file
```bash
llama stack run orchestrator_api.yaml
```

9. Go through the notebook to see how to use the stack, e.g. in an other terminal open:
```bash
jupyter notebook notebook-orchestrator-api.ipynb
```