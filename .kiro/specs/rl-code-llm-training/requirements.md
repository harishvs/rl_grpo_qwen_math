# Requirements Document

## Introduction

This document specifies requirements for a Reinforcement Learning (RL) training loop using GRPO (Group Relative Policy Optimization) via veRL on Amazon EKS. The system trains a Qwen model to solve basic grade-school math problems (GSM8K-style) using verifiable rewards.

The architecture separates GPU-intensive model training from CPU-based environment/reward computation across dedicated EKS node groups.

### Reference Implementation

The implementation will use the veRL example code as a starting point:
- **veRL GRPO Trainer Example**: https://github.com/verl-project/verl/blob/main/examples/grpo_trainer/run_qwen2_5_vl-7b.sh

This example provides the baseline configuration for running GRPO training with Qwen models, which will be adapted for our EKS deployment with separated GPU/CPU workloads.

## Glossary

- **GRPO**: Group Relative Policy Optimization - RL algorithm that computes advantages relative to a group of rollouts, eliminating the need for a value function
- **veRL**: Volcengine Reinforcement Learning framework for distributed LLM training with FSDP support
- **Trainer/Actor**: The component running GRPO updates on the policy model (GPU)
- **Reference Model**: Frozen copy of the initial policy for KL divergence computation (GPU)
- **Rollout**: Complete generation (problem → reasoning → answer) from the policy model
- **Environment Worker**: CPU service that evaluates model outputs and computes scalar rewards
- **Trajectory**: A complete sequence of (prompt, reasoning, answer, reward) for one problem

## Requirements

### Requirement 1: EKS Cluster Infrastructure

**User Story:** As an ML engineer, I want to run the training workload on Amazon EKS with dedicated GPU and CPU node groups, so that I can efficiently separate model training from environment computation.

#### Acceptance Criteria

1. THE EKS cluster SHALL be deployed in a single AWS region
2. THE cluster SHALL have a GPU node group using g5.48xlarge instances for trainer, actor, reference, and rollout generation
3. THE cluster SHALL have a CPU node group using c7i instances for environment, tool-calling, and reward computation
4. THE cluster networking SHALL allow communication between GPU and CPU nodes over the internal network
5. THE cluster SHALL support DNS resolution for CPU services from GPU pods

### Requirement 2: GPU Workload (Trainer/Actor/Reference)

**User Story:** As an ML engineer, I want the GRPO training loop to run entirely on GPU nodes, so that I can maximize training throughput with dedicated accelerator resources.

#### Acceptance Criteria

1. THE Trainer/Actor/Reference components SHALL run exclusively on the g5.48xlarge node group
2. THE system SHALL execute the GRPO algorithm via veRL, including actor updates and reference model usage
3. THE system SHALL handle all forward and backward passes for the Qwen model on GPU
4. THE GPU side SHALL continuously generate trajectories (problem → reasoning → answer) from the Qwen model
5. THE GPU side SHALL send trajectories to the CPU environment for reward calculation
6. THE GPU side SHALL apply GRPO updates to model parameters based on returned rewards

### Requirement 3: CPU Workload (Environment/Reward)

**User Story:** As an ML engineer, I want environment logic and reward computation to run on CPU nodes as a remote service, so that GPU resources remain dedicated to model training.

#### Acceptance Criteria

1. THE Environment/Tools/Reward components SHALL run exclusively on the c7i node group
2. THE environment SHALL accept prompts and model completions from the trainer/actor side
3. THE environment SHALL perform game/environment logic and compute scalar rewards indicating answer quality
4. THE trainer SHALL treat the environment as a remote service and SHALL NOT embed environment logic inside GPU nodes
5. THE CPU side SHALL accept requests, compute rewards deterministically, and respond within a bounded time per request
6. THE CPU side SHALL be horizontally scalable (multiple environment workers can be added without changing trainer logic)

### Requirement 4: Qwen Model Training

**User Story:** As an ML engineer, I want to fine-tune a Qwen model using GRPO to solve math word problems, so that the model learns step-by-step reasoning through reinforcement learning.

#### Acceptance Criteria

1. THE policy model SHALL be a small Qwen variant (e.g., Qwen3-1.7B)
2. THE model SHALL be fine-tuned to generate step-by-step solutions to math word problems
3. THE training loop SHALL use math word problems as input tasks (GSM8K-style)
4. THE system SHALL use the Qwen model as the policy being optimized with GRPO via veRL

### Requirement 5: Reward Computation

**User Story:** As an ML engineer, I want rewards to reflect answer correctness and format compliance, so that the model learns to produce both accurate and well-structured solutions.

#### Acceptance Criteria

1. THE reward function SHALL evaluate whether the final numeric answer is correct
2. THE reward function SHALL evaluate whether the output is parsable and follows the expected answer format
3. THE environment SHALL return scalar rewards for each trajectory
4. WHEN computing rewards THEN the environment SHALL operate deterministically (same input produces same reward)

### Requirement 6: Scalability and Reliability

**User Story:** As an ML engineer, I want the system to scale environment workers independently and handle transient failures gracefully, so that training can proceed reliably at scale.

#### Acceptance Criteria

1. THE design SHALL allow scaling the number of environment workers on CPU nodes without changing trainer logic
2. WHEN transient failures occur in environment workers THEN the system SHALL NOT crash the entire training
3. WHEN evaluations fail THEN they MAY be retried or skipped according to training policy
4. GPU resources SHALL be reserved for model training/inference only
5. Environment logic and tools SHALL be confined to CPU resources

### Requirement 7: Training Loop Execution

**User Story:** As an ML engineer, I want the end-to-end RL loop to execute correctly on EKS, so that I can train the model through repeated interaction with the environment.

#### Acceptance Criteria

1. THE training loop SHALL repeatedly sample math problems as prompts
2. THE training loop SHALL let the model generate reasoning and answers
3. THE training loop SHALL evaluate answers in an environment that computes numeric rewards
4. THE trainer SHALL repeatedly send math problems and model outputs to the environment
5. THE trainer SHALL update the model based on returned rewards via GRPO
6. Training SHALL complete multiple GRPO epochs on the EKS cluster

## Acceptance Criteria (System-Level)

The setup is considered successful when:

1. **Infrastructure**: Training completes multiple GRPO epochs with trainer/actor/reference on g5.48xlarge and environment/reward on c7i node groups

2. **End-to-End Loop**: The trainer repeatedly sends math problems and model outputs to the environment, receives scalar rewards, and updates the model via GRPO

3. **Learning Outcome**: The fine-tuned Qwen model shows measurable improvement in accuracy on held-out basic math word problems compared to the initial model

4. **Reasoning Quality**: The model's outputs contain coherent step-by-step reasoning aligned with correct solutions for basic math tasks (multi-step arithmetic word problems)

5. **Operational Success**: All components run as scheduled workloads in EKS without resource exhaustion or persistent connectivity issues
