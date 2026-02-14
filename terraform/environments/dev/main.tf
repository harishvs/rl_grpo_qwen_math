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
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name, "--output", "json"]
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
  gpu_disk_size      = var.gpu_disk_size

  # CPU node group settings
  cpu_instance_types = var.cpu_instance_types
  cpu_desired_size   = var.cpu_desired_size
  cpu_min_size       = var.cpu_min_size
  cpu_max_size       = var.cpu_max_size
  cpu_disk_size      = var.cpu_disk_size

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


# ECR Repositories
resource "aws_ecr_repository" "trainer" {
  name                 = "${var.project_name}-${var.environment}/trainer"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-trainer"
  })
}

resource "aws_ecr_repository" "environment" {
  name                 = "${var.project_name}-${var.environment}/environment"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-environment"
  })
}

# ECR Lifecycle Policy - keep last 10 images
resource "aws_ecr_lifecycle_policy" "trainer" {
  repository = aws_ecr_repository.trainer.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

resource "aws_ecr_lifecycle_policy" "environment" {
  repository = aws_ecr_repository.environment.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = {
          type = "expire"
        }
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

# ConfigMap with image URLs for Kubernetes manifests
resource "kubernetes_config_map" "image_config" {
  metadata {
    name      = "image-config"
    namespace = "default"
    labels = {
      app       = "grpo-trainer"
      component = "configuration"
    }
  }

  data = {
    trainer_image     = "${aws_ecr_repository.trainer.repository_url}:latest"
    environment_image = "${aws_ecr_repository.environment.repository_url}:latest"
  }
}

# Training ConfigMap with hyperparameters
resource "kubernetes_config_map" "training_config" {
  metadata {
    name      = "training-config"
    namespace = "default"
    labels = {
      app       = "grpo-trainer"
      component = "configuration"
    }
  }

  data = {
    model_name    = var.model_name
    batch_size    = tostring(var.batch_size)
    num_epochs    = tostring(var.num_epochs)
    group_size    = tostring(var.group_size)
    learning_rate = tostring(var.learning_rate)
    kl_coef       = tostring(var.kl_coef)
    clip_range    = tostring(var.clip_range)
  }
}

# Environment Service Deployment
resource "kubernetes_deployment" "environment_service" {
  metadata {
    name      = "environment-service"
    namespace = "default"
    labels = {
      app       = "environment-service"
      component = "reward-computation"
    }
  }

  spec {
    replicas = 1

    selector {
      match_labels = {
        app = "environment-service"
      }
    }

    template {
      metadata {
        labels = {
          app       = "environment-service"
          component = "reward-computation"
        }
      }

      spec {
        node_selector = {
          "node-type" = "cpu"
        }

        container {
          name  = "environment"
          image = "${aws_ecr_repository.environment.repository_url}:latest"

          port {
            container_port = 8080
            name           = "http"
            protocol       = "TCP"
          }

          resources {
            requests = {
              cpu    = "1"
              memory = "2Gi"
            }
            limits = {
              cpu    = "2"
              memory = "4Gi"
            }
          }

          liveness_probe {
            http_get {
              path = "/health"
              port = 8080
            }
            initial_delay_seconds = 10
            period_seconds        = 30
            timeout_seconds       = 10
            failure_threshold     = 3
          }

          readiness_probe {
            http_get {
              path = "/health"
              port = 8080
            }
            initial_delay_seconds = 5
            period_seconds        = 10
            timeout_seconds       = 5
            failure_threshold     = 3
          }

          env {
            name  = "PYTHONUNBUFFERED"
            value = "1"
          }
        }

        restart_policy                  = "Always"
        termination_grace_period_seconds = 30
      }
    }
  }

  depends_on = [module.node_groups]
}

# Environment Service
resource "kubernetes_service" "environment_service" {
  metadata {
    name      = "environment-service"
    namespace = "default"
    labels = {
      app       = "environment-service"
      component = "reward-computation"
    }
  }

  spec {
    selector = {
      app = "environment-service"
    }

    port {
      port        = 8080
      target_port = 8080
      protocol    = "TCP"
      name        = "http"
    }

    type = "ClusterIP"
  }
}

# GRPO Trainer Job
resource "kubernetes_job" "grpo_trainer" {
  metadata {
    name      = "grpo-trainer"
    namespace = "default"
    labels = {
      app       = "grpo-trainer"
      component = "model-training"
    }
  }

  spec {
    backoff_limit = 3

    template {
      metadata {
        labels = {
          app       = "grpo-trainer"
          component = "model-training"
        }
      }

      spec {
        node_selector = {
          "node-type" = "gpu"
        }

        service_account_name = kubernetes_service_account.grpo_trainer.metadata[0].name
        restart_policy       = "OnFailure"

        container {
          name  = "trainer"
          image = "${aws_ecr_repository.trainer.repository_url}:latest"

          resources {
            requests = {
              cpu               = "32"
              memory            = "256Gi"
              "nvidia.com/gpu"  = "8"
            }
            limits = {
              cpu               = "96"
              memory            = "384Gi"
              "nvidia.com/gpu"  = "8"
            }
          }

          env {
            name = "MODEL_NAME"
            value_from {
              config_map_key_ref {
                name = kubernetes_config_map.training_config.metadata[0].name
                key  = "model_name"
              }
            }
          }

          env {
            name  = "ENVIRONMENT_SERVICE_URL"
            value = "http://environment-service:8080"
          }

          env {
            name  = "CHECKPOINT_DIR"
            value = "s3://${aws_s3_bucket.checkpoints.id}/checkpoints"
          }

          env {
            name = "BATCH_SIZE"
            value_from {
              config_map_key_ref {
                name = kubernetes_config_map.training_config.metadata[0].name
                key  = "batch_size"
              }
            }
          }

          env {
            name = "NUM_EPOCHS"
            value_from {
              config_map_key_ref {
                name = kubernetes_config_map.training_config.metadata[0].name
                key  = "num_epochs"
              }
            }
          }

          env {
            name = "GROUP_SIZE"
            value_from {
              config_map_key_ref {
                name = kubernetes_config_map.training_config.metadata[0].name
                key  = "group_size"
              }
            }
          }

          env {
            name = "LEARNING_RATE"
            value_from {
              config_map_key_ref {
                name = kubernetes_config_map.training_config.metadata[0].name
                key  = "learning_rate"
              }
            }
          }

          env {
            name = "KL_COEF"
            value_from {
              config_map_key_ref {
                name = kubernetes_config_map.training_config.metadata[0].name
                key  = "kl_coef"
              }
            }
          }

          env {
            name = "CLIP_RANGE"
            value_from {
              config_map_key_ref {
                name = kubernetes_config_map.training_config.metadata[0].name
                key  = "clip_range"
              }
            }
          }

          env {
            name  = "AWS_DEFAULT_REGION"
            value = var.aws_region
          }

          volume_mount {
            name       = "shm"
            mount_path = "/dev/shm"
          }
        }

        volume {
          name = "shm"
          empty_dir {
            medium     = "Memory"
            size_limit = "64Gi"
          }
        }

        termination_grace_period_seconds = 300
      }
    }
  }

  wait_for_completion = false

  depends_on = [
    kubernetes_deployment.environment_service,
    kubernetes_service.environment_service,
    module.node_groups
  ]
}

# NVIDIA Device Plugin DaemonSet for GPU support
resource "kubernetes_daemonset" "nvidia_device_plugin" {
  metadata {
    name      = "nvidia-device-plugin-daemonset"
    namespace = "kube-system"
    labels = {
      app = "nvidia-device-plugin"
    }
  }

  spec {
    selector {
      match_labels = {
        app = "nvidia-device-plugin"
      }
    }

    template {
      metadata {
        labels = {
          app = "nvidia-device-plugin"
        }
      }

      spec {
        priority_class_name = "system-node-critical"
        
        toleration {
          key      = "nvidia.com/gpu"
          operator = "Exists"
          effect   = "NoSchedule"
        }

        node_selector = {
          "node-type" = "gpu"
        }

        container {
          name  = "nvidia-device-plugin-ctr"
          image = "nvcr.io/nvidia/k8s-device-plugin:v0.14.1"

          env {
            name  = "FAIL_ON_INIT_ERROR"
            value = "false"
          }

          security_context {
            allow_privilege_escalation = false
            capabilities {
              drop = ["ALL"]
            }
          }

          volume_mount {
            name       = "device-plugin"
            mount_path = "/var/lib/kubelet/device-plugins"
          }
        }

        volume {
          name = "device-plugin"
          host_path {
            path = "/var/lib/kubelet/device-plugins"
          }
        }
      }
    }

    strategy {
      type = "RollingUpdate"
    }
  }

  depends_on = [module.node_groups]
}
