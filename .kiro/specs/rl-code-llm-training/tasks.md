# Implementation Plan: RL Code LLM Training on EKS

## Overview

This implementation plan breaks down the GRPO training system into discrete coding tasks. The approach is:
1. Set up Terraform infrastructure first (EKS cluster, node groups)
2. Implement the CPU-side environment/reward service
3. Implement the GPU-side training components using veRL
4. Wire everything together with Kubernetes manifests
5. Validate with end-to-end tests

## Tasks

- [x] 1. Set up Terraform infrastructure foundation
  - [x] 1.1 Create Terraform backend configuration with S3 and DynamoDB
    - Create `terraform/backend.tf` with S3 backend for state storage
    - Create `terraform/environments/dev/backend.hcl` for backend config
    - _Requirements: 8.4_
  
  - [x] 1.2 Create VPC module
    - Create `terraform/modules/vpc/main.tf` with VPC, subnets, NAT gateway
    - Create `terraform/modules/vpc/variables.tf` and `outputs.tf`
    - Include private subnets for EKS nodes, public subnets for NAT
    - _Requirements: 8.2, 8.3_
  
  - [x] 1.3 Create EKS cluster module
    - Create `terraform/modules/eks/main.tf` with EKS cluster resource
    - Configure cluster IAM role and security groups
    - Enable cluster logging and OIDC provider
    - _Requirements: 8.2, 8.3, 1.1_
  
  - [x] 1.4 Create node groups module
    - Create `terraform/modules/node-groups/main.tf`
    - Define GPU node group (g5.48xlarge) with `node-type: gpu` label
    - Define CPU node group (c7i.large) with `node-type: cpu` label
    - Configure node IAM roles and instance profiles
    - _Requirements: 8.2, 8.3, 1.2, 1.3_
  
  - [x] 1.5 Create dev environment configuration
    - Create `terraform/environments/dev/main.tf` composing all modules
    - Create `terraform/environments/dev/variables.tf` and `terraform.tfvars`
    - Configure AWS region: us-east-1
    - Apply common tags for cost allocation
    - _Requirements: 8.1, 8.5, 8.6, 8.7, 1.1_

- [x] 2. Checkpoint - Validate Terraform infrastructure
  - Run `terraform init`, `terraform validate`, `terraform plan`
  - Ensure all resources are correctly defined
  - Ensure all tests pass, ask the user if questions arise.

- [-] 3. Implement reward computation service (CPU side)
  - [x] 3.1 Create reward computation core logic
    - Create `src/environment/reward.py` with `RewardWorker` class
    - Implement `extract_answer()` to parse numeric answers from completions
    - Implement `check_format()` to score format compliance (0.0-0.2)
    - Implement `compute_reward()` combining correctness and format scores
    - _Requirements: 5.1, 5.2, 5.3_
  
  - [x] 3.2 Write property test for reward determinism
    - **Property 2: Reward Determinism**
    - Generate random trajectories, verify identical rewards on repeated calls
    - **Validates: Requirements 5.4**
  
  - [x] 3.3 Write property test for reward correctness
    - **Property 3: Reward Correctness**
    - Generate completions with known answers, verify correctness_score is binary
    - Verify total reward is in valid range [0.0, 1.2]
    - **Validates: Requirements 5.1, 5.2, 5.3**
  
  - [x] 3.4 Create FastAPI environment service
    - Create `src/environment/service.py` with FastAPI app
    - Implement `/compute_rewards` endpoint accepting `RewardRequest`
    - Implement `/health` endpoint for Kubernetes probes
    - Add request timeout handling and error responses
    - _Requirements: 3.2, 3.3, 3.5_
  
  - [x] 3.5 Write property test for environment service contract
    - **Property 4: Environment Service Contract**
    - Generate random RewardRequests, verify response length matches request
    - **Validates: Requirements 3.2, 3.3, 3.5**
  
  - [x] 3.6 Create Dockerfile for environment service
    - Create `docker/environment/Dockerfile`
    - Use Python slim base image
    - Install FastAPI, uvicorn, and dependencies
    - _Requirements: 3.1_

- [x] 4. Checkpoint - Validate reward service
  - Run unit tests and property tests for reward computation
  - Test service locally with sample requests
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement GRPO training components (GPU side)
  - [x] 5.1 Create data models and configuration
    - Create `src/trainer/config.py` with `TrainingConfig`, `FSDPConfig` dataclasses
    - Create `src/trainer/models.py` with `Rollout`, `Trajectory`, `RewardRequest/Response`
    - _Requirements: 4.1, 4.3_
  
  - [x] 5.2 Create dataset loader for GSM8K-style problems
    - Create `src/trainer/dataset.py` with `MathProblemDataset` class
    - Implement prompt formatting for Qwen model
    - Support loading from HuggingFace datasets or local files
    - _Requirements: 4.3, 7.1_
  
  - [x] 5.3 Implement GRPO advantage computation
    - Create `src/trainer/grpo.py` with advantage calculation logic
    - Implement group normalization: `A_i = (r_i - mean(R)) / std(R)`
    - Handle edge case where std = 0 (all rewards identical)
    - _Requirements: 2.6, 7.5_
  
  - [x] 5.4 Write property test for GRPO advantage normalization
    - **Property 5: GRPO Advantage Normalization**
    - Generate random reward groups, verify mean ≈ 0 and std ≈ 1
    - **Validates: Requirements 2.6, 7.5**
  
  - [x] 5.5 Implement environment client
    - Create `src/trainer/environment_client.py`
    - Implement async HTTP client for environment service
    - Add retry logic with exponential backoff
    - Add timeout handling
    - _Requirements: 2.5, 3.4, 6.2, 6.3_
  
  - [x] 5.6 Implement veRL-based trainer
    - Create `src/trainer/trainer.py` with `GRPOTrainer` class
    - Integrate with veRL's `RayPPOTrainer` pattern
    - Configure FSDP for actor model, CPU offload for reference model
    - Implement `train_step()` orchestrating rollout → reward → update
    - _Requirements: 2.2, 2.3, 2.4, 2.6, 4.2, 4.4, 7.2, 7.3, 7.4, 7.5_
  
  - [x] 5.7 Write property test for training loop round-trip
    - **Property 1: Training Loop Round-Trip**
    - Mock environment service, verify trajectory/reward count invariants
    - **Validates: Requirements 2.4, 2.5, 2.6, 7.1, 7.2, 7.3, 7.4, 7.5**
  
  - [x] 5.8 Create training entrypoint script
    - Create `src/trainer/main.py` with CLI for training
    - Support configuration via environment variables and config files
    - Implement checkpoint saving to S3
    - _Requirements: 7.6_
  
  - [x] 5.9 Create Dockerfile for trainer
    - Create `docker/trainer/Dockerfile`
    - Use NVIDIA CUDA base image with PyTorch
    - Install veRL, vLLM, transformers, and dependencies
    - _Requirements: 2.1_

- [x] 6. Checkpoint - Validate training components
  - Run unit tests and property tests for GRPO logic
  - Test trainer initialization locally (without full cluster)
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Create Kubernetes manifests
  - [x] 7.1 Create environment service deployment and service
    - Create `k8s/environment/deployment.yaml` with nodeSelector for CPU nodes
    - Create `k8s/environment/service.yaml` with ClusterIP service
    - Configure resource requests/limits for c7i.large
    - Add health check probes
    - _Requirements: 3.1, 3.6, 1.4, 1.5_
  
  - [x] 7.2 Create trainer job manifest
    - Create `k8s/trainer/job.yaml` with nodeSelector for GPU nodes
    - Configure resource requests for g5.48xlarge (8 GPUs)
    - Mount S3 credentials for checkpoint saving
    - Set environment variables for environment service URL
    - _Requirements: 2.1, 1.4, 1.5_
  
  - [x] 7.3 Create ConfigMaps and Secrets
    - Create `k8s/config/training-config.yaml` with training hyperparameters
    - Create `k8s/config/secrets.yaml` template for AWS credentials
    - _Requirements: 4.1_
  
  - [x] 7.4 Write property test for node placement invariant
    - **Property 6: Node Placement Invariant**
    - Verify pod specs have correct nodeSelectors
    - **Validates: Requirements 2.1, 3.1, 6.4, 6.5**

- [ ] 8. Checkpoint - Validate Kubernetes manifests
  - Run `kubectl apply --dry-run=client` on all manifests
  - Verify nodeSelectors and resource requests are correct
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Integration and end-to-end wiring
  - [ ] 9.1 Create deployment scripts
    - Create `scripts/deploy-infrastructure.sh` for Terraform apply
    - Create `scripts/deploy-services.sh` for kubectl apply
    - Create `scripts/run-training.sh` to launch training job
    - _Requirements: 8.7_
  
  - [ ] 9.2 Create integration test suite
    - Create `tests/integration/test_e2e.py`
    - Test environment service reachability from trainer pod
    - Test single training step execution
    - _Requirements: 7.6_
  
  - [ ] 9.3 Add monitoring and logging configuration
    - Create `k8s/monitoring/` with basic CloudWatch agent config
    - Configure trainer to log metrics (loss, reward, KL divergence)
    - _Requirements: 6.2_

- [ ] 10. Final checkpoint - End-to-end validation
  - Deploy infrastructure with Terraform
  - Deploy services to EKS
  - Run short training job (1 epoch)
  - Verify model checkpoint saved to S3
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
- The implementation uses Python for both trainer and environment service
- veRL framework handles distributed training orchestration
