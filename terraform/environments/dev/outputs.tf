# Dev Environment Outputs

output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "private_subnet_ids" {
  description = "List of private subnet IDs"
  value       = module.vpc.private_subnet_ids
}

output "public_subnet_ids" {
  description = "List of public subnet IDs"
  value       = module.vpc.public_subnet_ids
}

output "cluster_name" {
  description = "Name of the EKS cluster"
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "Endpoint for the EKS cluster API server"
  value       = module.eks.cluster_endpoint
}

output "cluster_certificate_authority" {
  description = "Certificate authority data for the EKS cluster"
  value       = module.eks.cluster_certificate_authority
  sensitive   = true
}

output "gpu_node_group_arn" {
  description = "ARN of the GPU node group"
  value       = module.node_groups.gpu_node_group_arn
}

output "cpu_node_group_arn" {
  description = "ARN of the CPU node group"
  value       = module.node_groups.cpu_node_group_arn
}

output "configure_kubectl" {
  description = "Command to configure kubectl"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
}

output "checkpoints_bucket_name" {
  description = "Name of the S3 bucket for model checkpoints"
  value       = aws_s3_bucket.checkpoints.id
}

output "checkpoints_bucket_arn" {
  description = "ARN of the S3 bucket for model checkpoints"
  value       = aws_s3_bucket.checkpoints.arn
}

output "grpo_trainer_role_arn" {
  description = "ARN of the IAM role for GRPO trainer service account"
  value       = aws_iam_role.grpo_trainer.arn
}

# ECR Repository Outputs
output "trainer_ecr_repository_url" {
  description = "URL of the trainer ECR repository"
  value       = aws_ecr_repository.trainer.repository_url
}

output "environment_ecr_repository_url" {
  description = "URL of the environment service ECR repository"
  value       = aws_ecr_repository.environment.repository_url
}

# Docker build and push commands
output "docker_login_command" {
  description = "Command to login to ECR"
  value       = "aws ecr get-login-password --region ${var.aws_region} | docker login --username AWS --password-stdin ${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com"
}

output "build_trainer_image" {
  description = "Command to build and push trainer image"
  value       = "docker build -t ${aws_ecr_repository.trainer.repository_url}:latest -f docker/trainer/Dockerfile . && docker push ${aws_ecr_repository.trainer.repository_url}:latest"
}

output "build_environment_image" {
  description = "Command to build and push environment image"
  value       = "docker build -t ${aws_ecr_repository.environment.repository_url}:latest -f docker/environment/Dockerfile . && docker push ${aws_ecr_repository.environment.repository_url}:latest"
}

# FSx Lustre Outputs
output "fsx_lustre_id" {
  description = "ID of the FSx Lustre filesystem"
  value       = aws_fsx_lustre_file_system.training.id
}

output "fsx_lustre_dns_name" {
  description = "DNS name of the FSx Lustre filesystem"
  value       = aws_fsx_lustre_file_system.training.dns_name
}

output "fsx_lustre_mount_name" {
  description = "Mount name of the FSx Lustre filesystem"
  value       = aws_fsx_lustre_file_system.training.mount_name
}
