# Workload Resources Module Outputs

output "trainer_ecr_repository_url" {
  description = "URL of the trainer ECR repository"
  value       = aws_ecr_repository.trainer.repository_url
}

output "environment_ecr_repository_url" {
  description = "URL of the environment service ECR repository"
  value       = aws_ecr_repository.environment.repository_url
}

output "checkpoints_bucket_name" {
  description = "Name of the S3 bucket for checkpoints"
  value       = aws_s3_bucket.checkpoints.id
}

output "checkpoints_bucket_arn" {
  description = "ARN of the S3 bucket for checkpoints"
  value       = aws_s3_bucket.checkpoints.arn
}

output "trainer_role_arn" {
  description = "ARN of the IAM role for trainer IRSA"
  value       = aws_iam_role.trainer.arn
}

output "aws_account_id" {
  description = "AWS Account ID"
  value       = data.aws_caller_identity.current.account_id
}
