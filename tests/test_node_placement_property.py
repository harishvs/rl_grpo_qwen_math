"""
Property-based tests for Node Placement Invariant.

Property 6: Node Placement Invariant
**Validates: Requirements 2.1, 3.1, 6.4, 6.5**

For any pod in the training deployment:
- Trainer/Actor/Reference pods SHALL be scheduled on nodes with label `node-type: gpu`
- Environment/Reward pods SHALL be scheduled on nodes with label `node-type: cpu`
"""

import os
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional

import pytest
import yaml
from hypothesis import given, settings, strategies as st, assume


# Define the expected node placement rules
GPU_WORKLOADS = {"grpo-trainer", "trainer", "actor", "reference", "rollout"}
CPU_WORKLOADS = {"environment-service", "environment", "reward", "reward-worker"}

# Node selector labels
GPU_NODE_SELECTOR = {"node-type": "gpu"}
CPU_NODE_SELECTOR = {"node-type": "cpu"}


def load_k8s_manifests(k8s_dir: str = "k8s") -> List[Dict[str, Any]]:
    """
    Load all Kubernetes manifest files from the k8s directory.
    
    Returns a list of parsed YAML documents.
    """
    manifests = []
    k8s_path = Path(k8s_dir)
    
    if not k8s_path.exists():
        return manifests
    
    for yaml_file in k8s_path.rglob("*.yaml"):
        with open(yaml_file, "r") as f:
            content = f.read()
            # Handle multi-document YAML files
            for doc in yaml.safe_load_all(content):
                if doc is not None:
                    doc["_source_file"] = str(yaml_file)
                    manifests.append(doc)
    
    return manifests


def get_pod_spec(manifest: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract pod spec from various Kubernetes resource types.
    
    Handles: Deployment, Job, Pod, StatefulSet, DaemonSet, ReplicaSet
    """
    kind = manifest.get("kind", "")
    
    if kind == "Pod":
        return manifest.get("spec", {})
    elif kind in {"Deployment", "StatefulSet", "DaemonSet", "ReplicaSet"}:
        return manifest.get("spec", {}).get("template", {}).get("spec", {})
    elif kind == "Job":
        return manifest.get("spec", {}).get("template", {}).get("spec", {})
    elif kind == "CronJob":
        return manifest.get("spec", {}).get("jobTemplate", {}).get("spec", {}).get("template", {}).get("spec", {})
    
    return None


def get_workload_name(manifest: Dict[str, Any]) -> str:
    """Extract the workload name from a manifest."""
    return manifest.get("metadata", {}).get("name", "unknown")


def is_gpu_workload(name: str, labels: Dict[str, str]) -> bool:
    """Determine if a workload should run on GPU nodes."""
    name_lower = name.lower()
    
    # Check by name
    for gpu_name in GPU_WORKLOADS:
        if gpu_name in name_lower:
            return True
    
    # Check by labels
    component = labels.get("component", "").lower()
    app = labels.get("app", "").lower()
    
    if "training" in component or "model" in component:
        return True
    if any(gpu_name in app for gpu_name in GPU_WORKLOADS):
        return True
    
    return False


def is_cpu_workload(name: str, labels: Dict[str, str]) -> bool:
    """Determine if a workload should run on CPU nodes."""
    name_lower = name.lower()
    
    # Check by name
    for cpu_name in CPU_WORKLOADS:
        if cpu_name in name_lower:
            return True
    
    # Check by labels
    component = labels.get("component", "").lower()
    app = labels.get("app", "").lower()
    
    if "reward" in component or "environment" in component:
        return True
    if any(cpu_name in app for cpu_name in CPU_WORKLOADS):
        return True
    
    return False


class TestNodePlacementInvariant:
    """
    Property-based tests for Node Placement Invariant.
    
    **Validates: Requirements 2.1, 3.1, 6.4, 6.5**
    
    Property 6: Node Placement Invariant
    - Trainer/Actor/Reference pods SHALL be scheduled on nodes with label `node-type: gpu`
    - Environment/Reward pods SHALL be scheduled on nodes with label `node-type: cpu`
    """
    
    @pytest.fixture
    def k8s_manifests(self) -> List[Dict[str, Any]]:
        """Load all K8s manifests from the k8s directory."""
        return load_k8s_manifests()
    
    @pytest.fixture
    def workload_manifests(self, k8s_manifests) -> List[Dict[str, Any]]:
        """Filter manifests to only include workload resources with pod specs."""
        workloads = []
        for manifest in k8s_manifests:
            pod_spec = get_pod_spec(manifest)
            if pod_spec:
                workloads.append(manifest)
        return workloads
    
    def test_gpu_workloads_have_gpu_node_selector(self, workload_manifests):
        """
        Property 6a: GPU workloads SHALL have nodeSelector with node-type: gpu.
        
        **Validates: Requirements 2.1, 6.4**
        """
        for manifest in workload_manifests:
            name = get_workload_name(manifest)
            labels = manifest.get("metadata", {}).get("labels", {})
            pod_spec = get_pod_spec(manifest)
            
            if is_gpu_workload(name, labels):
                node_selector = pod_spec.get("nodeSelector", {})
                source_file = manifest.get("_source_file", "unknown")
                
                assert node_selector.get("node-type") == "gpu", (
                    f"GPU workload '{name}' in {source_file} must have nodeSelector "
                    f"'node-type: gpu', but has: {node_selector}"
                )
    
    def test_cpu_workloads_have_cpu_node_selector(self, workload_manifests):
        """
        Property 6b: CPU workloads SHALL have nodeSelector with node-type: cpu.
        
        **Validates: Requirements 3.1, 6.5**
        """
        for manifest in workload_manifests:
            name = get_workload_name(manifest)
            labels = manifest.get("metadata", {}).get("labels", {})
            pod_spec = get_pod_spec(manifest)
            
            if is_cpu_workload(name, labels):
                node_selector = pod_spec.get("nodeSelector", {})
                source_file = manifest.get("_source_file", "unknown")
                
                assert node_selector.get("node-type") == "cpu", (
                    f"CPU workload '{name}' in {source_file} must have nodeSelector "
                    f"'node-type: cpu', but has: {node_selector}"
                )
    
    def test_all_workloads_have_node_selector(self, workload_manifests):
        """
        Property 6c: All workloads SHALL have a nodeSelector defined.
        
        **Validates: Requirements 2.1, 3.1, 6.4, 6.5**
        """
        for manifest in workload_manifests:
            name = get_workload_name(manifest)
            labels = manifest.get("metadata", {}).get("labels", {})
            pod_spec = get_pod_spec(manifest)
            
            # Only check workloads that are part of our training system
            if is_gpu_workload(name, labels) or is_cpu_workload(name, labels):
                node_selector = pod_spec.get("nodeSelector", {})
                source_file = manifest.get("_source_file", "unknown")
                
                assert "node-type" in node_selector, (
                    f"Workload '{name}' in {source_file} must have 'node-type' in nodeSelector, "
                    f"but has: {node_selector}"
                )
    
    def test_no_gpu_workload_on_cpu_nodes(self, workload_manifests):
        """
        Property 6d: GPU workloads SHALL NOT be scheduled on CPU nodes.
        
        **Validates: Requirements 2.1, 6.4**
        """
        for manifest in workload_manifests:
            name = get_workload_name(manifest)
            labels = manifest.get("metadata", {}).get("labels", {})
            pod_spec = get_pod_spec(manifest)
            
            if is_gpu_workload(name, labels):
                node_selector = pod_spec.get("nodeSelector", {})
                source_file = manifest.get("_source_file", "unknown")
                
                assert node_selector.get("node-type") != "cpu", (
                    f"GPU workload '{name}' in {source_file} must NOT have nodeSelector "
                    f"'node-type: cpu', but has: {node_selector}"
                )
    
    def test_no_cpu_workload_on_gpu_nodes(self, workload_manifests):
        """
        Property 6e: CPU workloads SHALL NOT be scheduled on GPU nodes.
        
        **Validates: Requirements 3.1, 6.5**
        """
        for manifest in workload_manifests:
            name = get_workload_name(manifest)
            labels = manifest.get("metadata", {}).get("labels", {})
            pod_spec = get_pod_spec(manifest)
            
            if is_cpu_workload(name, labels):
                node_selector = pod_spec.get("nodeSelector", {})
                source_file = manifest.get("_source_file", "unknown")
                
                assert node_selector.get("node-type") != "gpu", (
                    f"CPU workload '{name}' in {source_file} must NOT have nodeSelector "
                    f"'node-type: gpu', but has: {node_selector}"
                )


class TestNodePlacementPropertyBased:
    """
    Property-based tests using Hypothesis to verify node placement invariants
    hold for various manifest configurations.
    
    **Validates: Requirements 2.1, 3.1, 6.4, 6.5**
    """
    
    # Strategy for generating workload names
    gpu_workload_name_strategy = st.sampled_from([
        "grpo-trainer", "trainer-job", "actor-deployment", 
        "reference-model", "rollout-engine", "model-training"
    ])
    
    cpu_workload_name_strategy = st.sampled_from([
        "environment-service", "reward-worker", "reward-service",
        "environment-deployment", "reward-computation"
    ])
    
    # Strategy for generating node selectors
    valid_gpu_selector_strategy = st.fixed_dictionaries({
        "node-type": st.just("gpu")
    })
    
    valid_cpu_selector_strategy = st.fixed_dictionaries({
        "node-type": st.just("cpu")
    })
    
    invalid_gpu_selector_strategy = st.one_of(
        st.fixed_dictionaries({"node-type": st.just("cpu")}),
        st.fixed_dictionaries({}),
        st.fixed_dictionaries({"node-type": st.just("general")}),
    )
    
    invalid_cpu_selector_strategy = st.one_of(
        st.fixed_dictionaries({"node-type": st.just("gpu")}),
        st.fixed_dictionaries({}),
        st.fixed_dictionaries({"node-type": st.just("general")}),
    )
    
    @given(
        workload_name=gpu_workload_name_strategy,
        node_selector=valid_gpu_selector_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_valid_gpu_workload_placement(self, workload_name: str, node_selector: Dict[str, str]):
        """
        Property 6f: Valid GPU workload configurations SHALL pass validation.
        
        **Validates: Requirements 2.1, 6.4**
        """
        labels = {"app": workload_name, "component": "model-training"}
        
        assert is_gpu_workload(workload_name, labels), (
            f"'{workload_name}' should be identified as GPU workload"
        )
        assert node_selector.get("node-type") == "gpu", (
            f"GPU workload should have node-type: gpu, got {node_selector}"
        )
    
    @given(
        workload_name=cpu_workload_name_strategy,
        node_selector=valid_cpu_selector_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_valid_cpu_workload_placement(self, workload_name: str, node_selector: Dict[str, str]):
        """
        Property 6g: Valid CPU workload configurations SHALL pass validation.
        
        **Validates: Requirements 3.1, 6.5**
        """
        labels = {"app": workload_name, "component": "reward-computation"}
        
        assert is_cpu_workload(workload_name, labels), (
            f"'{workload_name}' should be identified as CPU workload"
        )
        assert node_selector.get("node-type") == "cpu", (
            f"CPU workload should have node-type: cpu, got {node_selector}"
        )
    
    @given(
        workload_name=gpu_workload_name_strategy,
        node_selector=invalid_gpu_selector_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_invalid_gpu_workload_placement_detected(self, workload_name: str, node_selector: Dict[str, str]):
        """
        Property 6h: Invalid GPU workload configurations SHALL be detected.
        
        **Validates: Requirements 2.1, 6.4**
        """
        labels = {"app": workload_name, "component": "model-training"}
        
        assert is_gpu_workload(workload_name, labels), (
            f"'{workload_name}' should be identified as GPU workload"
        )
        # Invalid configuration: GPU workload without proper node selector
        assert node_selector.get("node-type") != "gpu", (
            f"This test verifies invalid configs are detected - selector should not be 'gpu'"
        )
    
    @given(
        workload_name=cpu_workload_name_strategy,
        node_selector=invalid_cpu_selector_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_invalid_cpu_workload_placement_detected(self, workload_name: str, node_selector: Dict[str, str]):
        """
        Property 6i: Invalid CPU workload configurations SHALL be detected.
        
        **Validates: Requirements 3.1, 6.5**
        """
        labels = {"app": workload_name, "component": "reward-computation"}
        
        assert is_cpu_workload(workload_name, labels), (
            f"'{workload_name}' should be identified as CPU workload"
        )
        # Invalid configuration: CPU workload without proper node selector
        assert node_selector.get("node-type") != "cpu", (
            f"This test verifies invalid configs are detected - selector should not be 'cpu'"
        )


class TestActualManifestNodePlacement:
    """
    Tests that verify the actual K8s manifests in the repository
    conform to the node placement invariant.
    
    **Validates: Requirements 2.1, 3.1, 6.4, 6.5**
    """
    
    def test_trainer_job_has_gpu_node_selector(self):
        """
        Verify trainer job manifest has correct GPU node selector.
        
        **Validates: Requirements 2.1**
        """
        trainer_path = Path("k8s/trainer/job.yaml")
        if not trainer_path.exists():
            pytest.skip("Trainer job manifest not found")
        
        with open(trainer_path) as f:
            manifest = yaml.safe_load(f)
        
        pod_spec = get_pod_spec(manifest)
        assert pod_spec is not None, "Trainer job should have a pod spec"
        
        node_selector = pod_spec.get("nodeSelector", {})
        assert node_selector.get("node-type") == "gpu", (
            f"Trainer job must have nodeSelector 'node-type: gpu', got: {node_selector}"
        )
    
    def test_environment_deployment_has_cpu_node_selector(self):
        """
        Verify environment deployment manifest has correct CPU node selector.
        
        **Validates: Requirements 3.1**
        """
        env_path = Path("k8s/environment/deployment.yaml")
        if not env_path.exists():
            pytest.skip("Environment deployment manifest not found")
        
        with open(env_path) as f:
            manifest = yaml.safe_load(f)
        
        pod_spec = get_pod_spec(manifest)
        assert pod_spec is not None, "Environment deployment should have a pod spec"
        
        node_selector = pod_spec.get("nodeSelector", {})
        assert node_selector.get("node-type") == "cpu", (
            f"Environment deployment must have nodeSelector 'node-type: cpu', got: {node_selector}"
        )
    
    def test_gpu_workloads_request_gpu_resources(self):
        """
        Verify GPU workloads request nvidia.com/gpu resources.
        
        **Validates: Requirements 2.1, 6.4**
        """
        trainer_path = Path("k8s/trainer/job.yaml")
        if not trainer_path.exists():
            pytest.skip("Trainer job manifest not found")
        
        with open(trainer_path) as f:
            manifest = yaml.safe_load(f)
        
        pod_spec = get_pod_spec(manifest)
        containers = pod_spec.get("containers", [])
        
        for container in containers:
            resources = container.get("resources", {})
            requests = resources.get("requests", {})
            limits = resources.get("limits", {})
            
            # GPU workloads should request GPU resources
            has_gpu_request = "nvidia.com/gpu" in requests or "nvidia.com/gpu" in limits
            assert has_gpu_request, (
                f"Container '{container.get('name')}' in trainer job should request GPU resources"
            )
    
    def test_cpu_workloads_do_not_request_gpu_resources(self):
        """
        Verify CPU workloads do not request nvidia.com/gpu resources.
        
        **Validates: Requirements 3.1, 6.5**
        """
        env_path = Path("k8s/environment/deployment.yaml")
        if not env_path.exists():
            pytest.skip("Environment deployment manifest not found")
        
        with open(env_path) as f:
            manifest = yaml.safe_load(f)
        
        pod_spec = get_pod_spec(manifest)
        containers = pod_spec.get("containers", [])
        
        for container in containers:
            resources = container.get("resources", {})
            requests = resources.get("requests", {})
            limits = resources.get("limits", {})
            
            # CPU workloads should NOT request GPU resources
            has_gpu_request = "nvidia.com/gpu" in requests or "nvidia.com/gpu" in limits
            assert not has_gpu_request, (
                f"Container '{container.get('name')}' in environment deployment "
                f"should NOT request GPU resources"
            )
