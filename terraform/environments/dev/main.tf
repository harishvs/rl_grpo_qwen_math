# Dev Environment - Main Configuration
# Composes VPC, EKS, and Node Groups modules

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.25"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = local.common_tags
  }
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority)

  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
  }
}

locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
    Team        = "ml-platform"
  }
}

# VPC Module
module "vpc" {
  source = "../../modules/vpc"

  project_name       = var.project_name
  environment        = var.environment
  vpc_cidr           = var.vpc_cidr
  availability_zones = var.availability_zones

  tags = local.common_tags
}

# EKS Cluster Module
module "eks" {
  source = "../../modules/eks"

  project_name       = var.project_name
  environment        = var.environment
  vpc_id             = module.vpc.vpc_id
  subnet_ids         = module.vpc.private_subnet_ids
  kubernetes_version = var.kubernetes_version

  tags = local.common_tags
}

# Node Groups Module
module "node_groups" {
  source = "../../modules/node-groups"

  project_name = var.project_name
  environment  = var.environment
  cluster_name = module.eks.cluster_name
  subnet_ids   = module.vpc.private_subnet_ids

  # GPU node group settings
  gpu_instance_types = var.gpu_instance_types
  gpu_desired_size   = var.gpu_desired_size
  gpu_min_size       = var.gpu_min_size
  gpu_max_size       = var.gpu_max_size

  # CPU node group settings
  cpu_instance_types = var.cpu_instance_types
  cpu_desired_size   = var.cpu_desired_size
  cpu_min_size       = var.cpu_min_size
  cpu_max_size       = var.cpu_max_size

  tags = local.common_tags
}

# S3 Bucket for Model Checkpoints
resource "aws_s3_bucket" "checkpoints" {
  bucket = "${var.project_name}-${var.environment}-checkpoints"

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-${var.environment}-checkpoints"
  })
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

# IAM Role for GRPO Trainer Service Account (IRSA)
data "aws_caller_identity" "current" {}

locals {
  oidc_provider_id = replace(module.eks.oidc_provider_url, "https://", "")
}

resource "aws_iam_role" "grpo_trainer" {
  name = "${var.project_name}-${var.environment}-grpo-trainer-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = module.eks.oidc_provider_arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${local.oidc_provider_id}:aud" = "sts.amazonaws.com"
            "${local.oidc_provider_id}:sub" = "system:serviceaccount:default:grpo-trainer-sa"
          }
        }
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy" "grpo_trainer_s3" {
  name = "s3-checkpoint-access"
  role = aws_iam_role.grpo_trainer.id

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
      }
    ]
  })
}


# Kubernetes Resources
# Service Account with IRSA annotation
resource "kubernetes_service_account" "grpo_trainer" {
  metadata {
    name      = "grpo-trainer-sa"
    namespace = "default"
    labels = {
      app       = "grpo-trainer"
      component = "identity"
    }
    annotations = {
      "eks.amazonaws.com/role-arn" = aws_iam_role.grpo_trainer.arn
    }
  }
}

# Training secrets with S3 bucket name
resource "kubernetes_secret" "training_secrets" {
  metadata {
    name      = "training-secrets"
    namespace = "default"
    labels = {
      app       = "grpo-trainer"
      component = "configuration"
    }
  }

  data = {
    s3_bucket = aws_s3_bucket.checkpoints.id
  }
}
