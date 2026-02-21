# Workload Resources Module
# Creates ECR repositories, S3 bucket for checkpoints, and IRSA role

# ECR Repository for Trainer
resource "aws_ecr_repository" "trainer" {
  name                 = "${var.project_name}/trainer"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = var.tags
}

# ECR Repository for Environment Service
resource "aws_ecr_repository" "environment" {
  name                 = "${var.project_name}/environment"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = var.tags
}

# S3 Bucket for Model Checkpoints
resource "aws_s3_bucket" "checkpoints" {
  bucket = "${var.project_name}-checkpoints-${data.aws_caller_identity.current.account_id}-${var.aws_region}"

  tags = var.tags
}

resource "aws_s3_bucket_versioning" "checkpoints" {
  bucket = aws_s3_bucket.checkpoints.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "checkpoints" {
  bucket = aws_s3_bucket.checkpoints.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "checkpoints" {
  bucket = aws_s3_bucket.checkpoints.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Data source for current AWS account
data "aws_caller_identity" "current" {}

# IAM Policy for trainer to access S3
resource "aws_iam_policy" "trainer_s3_access" {
  name        = "${var.project_name}-trainer-s3-access"
  description = "Allows GRPO trainer to access S3 checkpoints and push CloudWatch metrics"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.checkpoints.arn,
          "${aws_s3_bucket.checkpoints.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "cloudwatch:namespace" = "GRPO-Training"
          }
        }
      }
    ]
  })

  tags = var.tags
}

# IAM Role for IRSA (trainer service account)
resource "aws_iam_role" "trainer" {
  name = "${var.project_name}-trainer-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = var.oidc_provider_arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringLike = {
            "${var.oidc_provider_url}:sub" = "system:serviceaccount:${var.trainer_namespace}:*-trainer-sa"
          }
          StringEquals = {
            "${var.oidc_provider_url}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "trainer_s3_access" {
  role       = aws_iam_role.trainer.name
  policy_arn = aws_iam_policy.trainer_s3_access.arn
}
