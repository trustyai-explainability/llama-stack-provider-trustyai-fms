#!/bin/bash

NAMESPACE="test"
BASE_PATH="tests/kind/manifests"
VLLM_EMULATOR="vllm_emulator.yaml"
LLAMA_STACK_DISTRIBUTION="llama_stack_distribution.yaml"
ORCHESTRATOR_CONFIGMAP="gorch_cm.yaml"
GUARDRAILS_TLS_SECRET="gorch_cm.yaml"
GUARDRAILS_ORCHESTRATOR="gorch.yaml"

# Set the provider image for substitution
export PROVIDER_IMAGE=

# Function to wait for pods to be ready
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
        exit 1
    fi
    echo "$description is ready"
}

# Create a namespace for testing
kubectl create namespace "$NAMESPACE"

# Deploy the vLLM emulator
kubectl apply -f ${BASE_PATH}/${VLLM_EMULATOR} -n "$NAMESPACE"
wait_for_pods "$NAMESPACE" "app=vllm-emulator" 300 "vLLM emulator"

# Wait for the TrustyAI operator controller to be ready
echo "Waiting for TrustyAI operator controller to be ready..."
if ! kubectl wait --for=condition=ready pod -l control-plane=controller-manager -n system --timeout=300s; then
    echo "ERROR: TrustyAI operator controller failed to become ready within 300 seconds"
    kubectl get pods -n system
    kubectl logs -n system $(kubectl get pods -n system | grep trustyai-service-operator-controller-manager | awk '{print $1}') --tail=100
    exit 1
fi
echo "TrustyAI operator controller is ready"

# Deploy the orchestrator ConfigMap and the GuardrailsOrchestrator
kubectl apply -f ${BASE_PATH}/${ORCHESTRATOR_CONFIGMAP} -n "$NAMESPACE"
kubectl apply -f ${BASE_PATH}/${GUARDRAILS_TLS_SECRET} -n "$NAMESPACE"
kubectl apply -f ${BASE_PATH}/${GUARDRAILS_ORCHESTRATOR} -n "$NAMESPACE"

echo "=============================="
echo "MONITORING TRUSTYAI OPERATOR AND GUARDRAILS ORCHESTRATOR"
echo "=============================="

# Monitor for 3 minutes with 30-second intervals
for i in {1..6}; do
    echo "--- Check $i/6 ($(date)) ---"

    echo "TrustyAI Operator Pod Status:"
    kubectl get pods -n system -l control-plane=controller-manager

    echo "TrustyAI Operator Logs (last 20 lines):"
    kubectl logs -n system $(kubectl get pods -n system | grep trustyai-service-operator-controller-manager | awk '{print $1}') --tail=20 2>/dev/null || echo "No logs available"

    echo "GuardrailsOrchestrator Status:"
    kubectl get GuardrailsOrchestrator guardrails-orchestrator -n "$NAMESPACE" -o wide 2>/dev/null || echo "GuardrailsOrchestrator not found"

    echo "GuardrailsOrchestrator Description:"
    kubectl describe GuardrailsOrchestrator guardrails-orchestrator -n "$NAMESPACE" 2>/dev/null || echo "GuardrailsOrchestrator not found"

    echo "All resources in test namespace:"
    kubectl get all -n "$NAMESPACE"

    echo "Events in test namespace:"
    kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' | tail -10

    if [ $i -lt 6 ]; then
        echo "Waiting 30 seconds before next check..."
        sleep 30
    fi
    echo ""
done

echo "=============================="
echo "FINAL STATUS CHECK"
echo "=============================="

wait_for_pods "$NAMESPACE" "app.kubernetes.io/name=guardrails-orchestrator" 300 "GuardrailsOrchestrator"

# Deploy the LlamaStackDistribution with image substitution
envsubst < ${BASE_PATH}/${LLAMA_STACK_DISTRIBUTION} | kubectl apply -f - -n "$NAMESPACE"
wait_for_pods "llamastack" "app.kubernetes.io/name=llamastack-custom-distribution" 300 "LlamaStackDistribution"


