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
