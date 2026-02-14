# Dev Environment Variables

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "rl-code-llm-training"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

variable "kubernetes_version" {
  description = "Kubernetes version for EKS"
  type        = string
  default     = "1.29"
}

# GPU Node Group Settings
variable "gpu_instance_types" {
  description = "Instance types for GPU nodes"
  type        = list(string)
  default     = ["g5.48xlarge"]
}

variable "gpu_desired_size" {
  description = "Desired number of GPU nodes"
  type        = number
  default     = 1
}

variable "gpu_min_size" {
  description = "Minimum number of GPU nodes"
  type        = number
  default     = 0
}

variable "gpu_max_size" {
  description = "Maximum number of GPU nodes"
  type        = number
  default     = 2
}

variable "gpu_disk_size" {
  description = "Root volume size in GB for GPU nodes"
  type        = number
  default     = 100
}

# CPU Node Group Settings
variable "cpu_instance_types" {
  description = "Instance types for CPU nodes"
  type        = list(string)
  default     = ["c7i.large"]
}

variable "cpu_desired_size" {
  description = "Desired number of CPU nodes"
  type        = number
  default     = 2
}

variable "cpu_min_size" {
  description = "Minimum number of CPU nodes"
  type        = number
  default     = 1
}

variable "cpu_max_size" {
  description = "Maximum number of CPU nodes"
  type        = number
  default     = 10
}

variable "cpu_disk_size" {
  description = "Root volume size in GB for CPU nodes"
  type        = number
  default     = 50
}

# Training Hyperparameters
variable "model_name" {
  description = "Name of the model to train"
  type        = string
  default     = "Qwen/Qwen2.5-1.5B"
}

variable "batch_size" {
  description = "Training batch size"
  type        = number
  default     = 32
}

variable "num_epochs" {
  description = "Number of training epochs"
  type        = number
  default     = 3
}

variable "group_size" {
  description = "GRPO group size (rollouts per prompt)"
  type        = number
  default     = 8
}

variable "learning_rate" {
  description = "Learning rate for training"
  type        = number
  default     = 0.000001
}

variable "kl_coef" {
  description = "KL divergence coefficient"
  type        = number
  default     = 0.1
}

variable "clip_range" {
  description = "PPO clip range"
  type        = number
  default     = 0.2
}
