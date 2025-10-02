#!/bin/bash

NAMESPACE="test"
BASE_PATH="tests/kind/manifests"
VLLM_EMULATOR="vllm_emulator.yaml"
LLAMA_STACK_DISTRIBUTION="llama_stack_distribution.yaml"
ORCHESTRATOR_CONFIGMAP="gorch_cm.yaml"
GUARDRAILS_TLS_SECRET="gorch_tls.yaml"
GUARDRAILS_ORCHESTRATOR="gorch.yaml"


wait_for_pods() {
    local namespace=$1
    local label_selector=$2
    local timeout=${3:-300}
    local description=$4

    echo "Waiting for $description to be ready..."
    if ! kubectl wait --for=condition=ready pod -l "$label_selector" -n "$namespace" --timeout="${timeout}s"; then
        echo "ERROR: $description failed to become ready within ${timeout} seconds"
        kubectl get pods -l "$label_selector" -n "$namespace"
        kubectl describe pods -l "$label_selector" -n "$namespace"
        return 1
    fi
    echo "$description is ready"
    return 0
}

# Ensure that the TrustyAI operator controller is ready
echo "Waiting for TrustyAI operator controller to be ready..."
if ! kubectl wait --for=condition=ready pod -l control-plane=controller-manager -n system --timeout=300s; then
    echo "ERROR: TrustyAI operator controller failed to become ready within 300 seconds"
    kubectl get pods -n system
    kubectl logs -n system $(kubectl get pods -n system | grep trustyai-service-operator-controller-manager | awk '{print $1}') --tail=100
    exit 1
fi
echo "TrustyAI operator controller is ready"


# Create a namespace for testing
kubectl create namespace "$NAMESPACE"

# Deploy the vLLM emulator
kubectl apply -f ${BASE_PATH}/${VLLM_EMULATOR} -n "$NAMESPACE"
wait_for_pods "$NAMESPACE" "app=vllm-emulator" 300 "vLLM emulator"

# Deploy the orchestrator ConfigMap and the GuardrailsOrchestrator
kubectl apply -f ${BASE_PATH}/${ORCHESTRATOR_CONFIGMAP} -n "$NAMESPACE"
kubectl apply -f ${BASE_PATH}/${GUARDRAILS_TLS_SECRET} -n "$NAMESPACE"
kubectl apply -f ${BASE_PATH}/${GUARDRAILS_ORCHESTRATOR} -n "$NAMESPACE"

# Wait a moment for the GuardrailsOrchestrator deployment to be created
sleep 60

# Patch the deployment to remove runAsNonRoot security context at both pod and container level
echo "Patching GuardrailsOrchestrator deployment to remove runAsNonRoot security context..."
kubectl patch deployment guardrails-orchestrator -n "$NAMESPACE" --type='strategic' -p='{"spec":{"template":{"spec":{"securityContext":{"runAsNonRoot":false},"containers":[{"name":"guardrails-orchestrator","securityContext":{"runAsNonRoot":false}}]}}}}'

# Restart the rollout to apply the patch
echo "Restarting deployment rollout to apply security context changes..."/
kubectl rollout restart deployment/guardrails-orchestrator -n "$NAMESPACE"

sleep 60
# Ensure that the GuardrailsOrchestrator pods are ready
if ! wait_for_pods "$NAMESPACE" "app.kubernetes.io/name=guardrails-orchestrator" 120 "GuardrailsOrchestrator"; then
    echo "GuardrailsOrchestrator failed to start. Collecting logs for debugging..."
    kubectl logs -l app.kubernetes.io/name=guardrails-orchestrator -c guardrails-orchestrator -n "$NAMESPACE" --tail=50 || echo "No logs available"
    exit 1
fi

# Deploy the LlamaStackDistribution
envsubst < ${BASE_PATH}/${LLAMA_STACK_DISTRIBUTION} | kubectl apply -f - -n "$NAMESPACE"

# Check if the LlamaStackDistribution pods are ready
if ! wait_for_pods "test" "app.kubernetes.io/instance=llamastack-custom-distribution" 60 "LlamaStackDistribution"; then
    echo "LlamaStackDistribution failed to start. Collecting debugging information..."
    echo "Pod status:"
    kubectl get pods -l app.kubernetes.io/instance=llamastack-custom-distribution -n "$NAMESPACE"
    echo "Pod description:"
    kubectl describe pods -l app.kubernetes.io/instance=llamastack-custom-distribution -n "$NAMESPACE"
    echo "Pod logs:"
    kubectl logs -l app.kubernetes.io/instance=llamastack-custom-distribution -n "$NAMESPACE" --tail=50 --all-containers=true || echo "No logs available"
    exit 1
fi

# TODO: Register and run shield using the LlamaStack Client

