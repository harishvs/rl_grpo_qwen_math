# Dev Environment - Terraform Variables

aws_region   = "us-east-1"
project_name = "rl-code-llm-training"
environment  = "dev"

vpc_cidr           = "10.0.0.0/16"
availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]

kubernetes_version = "1.29"

# GPU Node Group (g5.48xlarge for trainer/actor/reference)
gpu_instance_types = ["g5.48xlarge"]
gpu_desired_size   = 1
gpu_min_size       = 0
gpu_max_size       = 2

# CPU Node Group (c7i.large for environment/reward workers)
cpu_instance_types = ["c7i.large"]
cpu_desired_size   = 2
cpu_min_size       = 1
cpu_max_size       = 10
