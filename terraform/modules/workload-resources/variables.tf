# Workload Resources Module Variables

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "rl-grpo-training"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
}

variable "oidc_provider_arn" {
  description = "ARN of the EKS OIDC provider for IRSA"
  type        = string
}

variable "oidc_provider_url" {
  description = "URL of the EKS OIDC provider (without https://)"
  type        = string
}

variable "trainer_namespace" {
  description = "Kubernetes namespace where trainer runs"
  type        = string
  default     = "default"
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {}
}
