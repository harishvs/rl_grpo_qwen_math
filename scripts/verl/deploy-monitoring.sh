#!/bin/bash
# Deploy Prometheus + Grafana monitoring stack on CPU nodes.
# Idempotent — safe to run multiple times.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
K8S_DIR="$(cd "$SCRIPT_DIR/../../k8s/verl" && pwd)"

echo "=== Deploying monitoring stack ==="

# StorageClass
kubectl apply -f "$K8S_DIR/storageclass.yaml"

# Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true
helm repo update >/dev/null

# Namespace
kubectl create namespace monitoring 2>/dev/null || true

# Install/upgrade kube-prometheus-stack
helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --set prometheus.prometheusSpec.nodeSelector."node\\.kubernetes\\.io/instance-type"=c7i.large \
  --set grafana.nodeSelector."node\\.kubernetes\\.io/instance-type"=c7i.large \
  --set alertmanager.enabled=false \
  --set prometheus.prometheusSpec.retention=7d \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.storageClassName=gp3 \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=10Gi \
  --set grafana.adminPassword=grpo-training \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
  --wait --timeout 5m

# PodMonitor + Grafana dashboard
kubectl apply -f "$K8S_DIR/podmonitor.yaml"
kubectl apply -f "$K8S_DIR/grafana-dashboard.yaml"

echo ""
echo "=== Monitoring deployed ==="
echo "Access Grafana:"
echo "  kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80"
echo "  http://localhost:3000 — admin / grpo-training"
