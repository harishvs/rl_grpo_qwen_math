# Node Groups Module Variables

variable "project_name" {
  description = "Name of the project, used for resource naming"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
}

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
}

variable "subnet_ids" {
  description = "List of subnet IDs for node groups"
  type        = list(string)
}

variable "gpu_subnet_ids" {
  description = "Subnet IDs for GPU node group (use to pin to specific AZ for capacity blocks). Defaults to subnet_ids if not set."
  type        = list(string)
  default     = []
}

variable "gpu_instance_types" {
  description = "Instance types for GPU node group"
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

variable "cpu_instance_types" {
  description = "Instance types for CPU node group"
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

variable "gpu_disk_size" {
  description = "Root volume size in GB for GPU nodes"
  type        = number
  default     = 100
}

variable "cpu_disk_size" {
  description = "Root volume size in GB for CPU nodes"
  type        = number
  default     = 50
}

variable "vpc_id" {
  description = "VPC ID for EFA security group"
  type        = string
}

variable "cluster_security_group_id" {
  description = "EKS cluster security group ID"
  type        = string
  default     = ""
}

variable "efa_enabled" {
  description = "Enable EFA network interfaces on GPU nodes"
  type        = bool
  default     = false
}

variable "capacity_reservation_id" {
  description = "EC2 Capacity Block reservation ID for GPU nodes (e.g. cr-xxxxx). Leave empty to use on-demand."
  type        = string
  default     = ""
}

variable "tags" {
  description = "Common tags to apply to all resources"
  type        = map(string)
  default     = {}
}
