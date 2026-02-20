# Dev Environment - Terraform Variables

aws_region   = "us-east-1"
project_name = "rl-code-llm-training"
environment  = "dev"

vpc_cidr           = "10.0.0.0/16"
availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]

kubernetes_version = "1.29"

# GPU Node Group (p4d.24xlarge for multi-node training)
gpu_instance_types = ["p4d.24xlarge"]
gpu_desired_size   = 2
gpu_min_size       = 0
gpu_max_size       = 2

# EFA for cross-node NCCL
efa_enabled             = true
capacity_reservation_id = "cr-REDACTED"
gpu_subnet_ids          = ["subnet-REDACTED"]  # us-east-1a only

# CPU Node Group (c7i.large for environment/reward workers)
cpu_instance_types = ["c7i.large"]
cpu_desired_size   = 2
cpu_min_size       = 1
cpu_max_size       = 10
