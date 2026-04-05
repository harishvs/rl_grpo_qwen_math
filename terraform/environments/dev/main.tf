# Dev Environment - Main Configuration
# Composes VPC, EKS, and Node Groups modules

terraform {
  required_version = ">= 1.0"

  backend "s3" {
    bucket         = "rl-code-llm-training-tfstate-dev"
    key            = "environments/dev/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "rl-code-llm-training-tflock-dev"
    encrypt        = true
  }

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
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.12"
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

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority)

    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name, "--output", "json"]
    }
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
  vpc_id       = module.vpc.vpc_id
  cluster_security_group_id = module.eks.cluster_security_group_id
  efa_enabled  = var.efa_enabled
  capacity_reservation_id = var.capacity_reservation_id
  gpu_subnet_ids = var.gpu_subnet_ids

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
          }
          StringLike = {
            "${local.oidc_provider_id}:sub" = "system:serviceaccount:default:*-trainer-sa"
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

resource "kubernetes_service_account" "verl_trainer" {
  metadata {
    name      = "verl-trainer-sa"
    namespace = "default"
    labels = {
      app       = "verl-trainer"
      component = "identity"
    }
    annotations = {
      "eks.amazonaws.com/role-arn" = aws_iam_role.grpo_trainer.arn
    }
  }
}


# =============================================================================
# FSx for Lustre - Shared storage for multi-node training
# =============================================================================

# Security group for FSx Lustre
resource "aws_security_group" "fsx_lustre" {
  name        = "${var.project_name}-${var.environment}-fsx-lustre-sg"
  description = "Security group for FSx Lustre filesystem"
  vpc_id      = module.vpc.vpc_id

  ingress {
    description = "Lustre traffic from VPC"
    from_port   = 988
    to_port     = 988
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  ingress {
    description = "Lustre traffic from VPC"
    from_port   = 1021
    to_port     = 1023
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-${var.environment}-fsx-lustre-sg"
  })
}

# FSx for Lustre filesystem
resource "aws_fsx_lustre_file_system" "training" {
  storage_capacity            = var.fsx_storage_capacity_gb
  subnet_ids                  = [module.vpc.private_subnet_ids[0]]
  security_group_ids          = [aws_security_group.fsx_lustre.id]
  deployment_type             = "SCRATCH_2"
  storage_type                = "SSD"
  file_system_type_version    = "2.15"

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-${var.environment}-fsx-lustre"
  })
}

# FSx CSI Driver IAM Role (IRSA)
resource "aws_iam_role" "fsx_csi_driver" {
  name = "${var.project_name}-${var.environment}-fsx-csi-driver-role"

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
            "${local.oidc_provider_id}:sub" = "system:serviceaccount:kube-system:fsx-csi-controller-sa"
          }
        }
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "fsx_csi_driver" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonFSxFullAccess"
  role       = aws_iam_role.fsx_csi_driver.name
}

# FSx CSI Driver EKS Addon
resource "aws_eks_addon" "fsx_csi_driver" {
  cluster_name             = module.eks.cluster_name
  addon_name               = "aws-fsx-csi-driver"
  service_account_role_arn = aws_iam_role.fsx_csi_driver.arn

  resolve_conflicts_on_create = "OVERWRITE"
  resolve_conflicts_on_update = "OVERWRITE"

  tags = local.common_tags

  depends_on = [module.node_groups]
}

# Kubernetes StorageClass for FSx Lustre
resource "kubernetes_storage_class" "fsx_lustre" {
  metadata {
    name = "fsx-lustre"
  }

  storage_provisioner = "fsx.csi.aws.com"

  parameters = {
    subnetId         = module.vpc.private_subnet_ids[0]
    securityGroupIds = aws_security_group.fsx_lustre.id
    deploymentType   = "SCRATCH_2"
    storageType      = "SSD"
  }

  reclaim_policy      = "Delete"
  volume_binding_mode = "Immediate"

  depends_on = [aws_eks_addon.fsx_csi_driver]
}

# Static PV for the FSx Lustre filesystem
resource "kubernetes_persistent_volume" "fsx_training" {
  metadata {
    name = "fsx-training-pv"
  }

  spec {
    capacity = {
      storage = "${var.fsx_storage_capacity_gb}Gi"
    }

    volume_mode                      = "Filesystem"
    access_modes                     = ["ReadWriteMany"]
    persistent_volume_reclaim_policy = "Retain"
    storage_class_name               = kubernetes_storage_class.fsx_lustre.metadata[0].name

    persistent_volume_source {
      csi {
        driver        = "fsx.csi.aws.com"
        volume_handle = aws_fsx_lustre_file_system.training.id

        volume_attributes = {
          "dnsname"   = aws_fsx_lustre_file_system.training.dns_name
          "mountname" = aws_fsx_lustre_file_system.training.mount_name
        }
      }
    }
  }

  depends_on = [aws_eks_addon.fsx_csi_driver]
}

# PVC for training checkpoints
resource "kubernetes_persistent_volume_claim" "fsx_training" {
  metadata {
    name      = "fsx-training-checkpoints"
    namespace = "default"
  }

  spec {
    access_modes       = ["ReadWriteMany"]
    storage_class_name = kubernetes_storage_class.fsx_lustre.metadata[0].name

    resources {
      requests = {
        storage = "${var.fsx_storage_capacity_gb}Gi"
      }
    }

    volume_name = kubernetes_persistent_volume.fsx_training.metadata[0].name
  }
}

# EBS CSI Driver IAM Role (IRSA)
resource "aws_iam_role" "ebs_csi_driver" {
  name = "${var.project_name}-${var.environment}-ebs-csi-driver-role"

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
            "${local.oidc_provider_id}:sub" = "system:serviceaccount:kube-system:ebs-csi-controller-sa"
          }
        }
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "ebs_csi_driver" {
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy"
  role       = aws_iam_role.ebs_csi_driver.name
}

# EBS CSI Driver EKS Addon
resource "aws_eks_addon" "ebs_csi_driver" {
  cluster_name             = module.eks.cluster_name
  addon_name               = "aws-ebs-csi-driver"
  service_account_role_arn = aws_iam_role.ebs_csi_driver.arn
  
  # Use the latest compatible version
  resolve_conflicts_on_create = "OVERWRITE"
  resolve_conflicts_on_update = "OVERWRITE"

  tags = local.common_tags

  depends_on = [module.node_groups]
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

# AWS EFA Device Plugin DaemonSet
resource "kubernetes_daemonset" "efa_device_plugin" {
  count = var.efa_enabled ? 1 : 0

  metadata {
    name      = "aws-efa-k8s-device-plugin-daemonset"
    namespace = "kube-system"
  }

  spec {
    selector {
      match_labels = {
        name = "aws-efa-k8s-device-plugin"
      }
    }

    template {
      metadata {
        labels = {
          name = "aws-efa-k8s-device-plugin"
        }
      }

      spec {
        priority_class_name = "system-node-critical"
        host_network        = true

        toleration {
          key      = "nvidia.com/gpu"
          operator = "Exists"
          effect   = "NoSchedule"
        }

        node_selector = {
          "node-type" = "gpu"
        }

        container {
          name  = "aws-efa-k8s-device-plugin"
          image = "602401143452.dkr.ecr.us-east-1.amazonaws.com/eks/aws-efa-k8s-device-plugin:v0.5.7"

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

# =============================================================================
# CloudWatch Container Insights and Fluent Bit Logging
# =============================================================================

# CloudWatch Log Group for container logs
resource "aws_cloudwatch_log_group" "container_logs" {
  name              = "/aws/eks/${var.project_name}-${var.environment}/containers"
  retention_in_days = 7

  tags = local.common_tags
}

# CloudWatch Log Group for application logs
resource "aws_cloudwatch_log_group" "application_logs" {
  name              = "/aws/eks/${var.project_name}-${var.environment}/application"
  retention_in_days = 14

  tags = local.common_tags
}

# IAM Role for Fluent Bit (IRSA)
resource "aws_iam_role" "fluent_bit" {
  name = "${var.project_name}-${var.environment}-fluent-bit-role"

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
            "${local.oidc_provider_id}:sub" = "system:serviceaccount:amazon-cloudwatch:fluent-bit"
          }
        }
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy" "fluent_bit_cloudwatch" {
  name = "cloudwatch-logs-access"
  role = aws_iam_role.fluent_bit.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:CreateLogGroup",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams",
          "logs:DescribeLogGroups"
        ]
        Resource = [
          "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/eks/${var.project_name}-${var.environment}/*",
          "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/eks/${var.project_name}-${var.environment}/*:*"
        ]
      }
    ]
  })
}

# Kubernetes namespace for CloudWatch
resource "kubernetes_namespace" "amazon_cloudwatch" {
  metadata {
    name = "amazon-cloudwatch"
    labels = {
      name = "amazon-cloudwatch"
    }
  }
}

# Service Account for Fluent Bit
resource "kubernetes_service_account" "fluent_bit" {
  metadata {
    name      = "fluent-bit"
    namespace = kubernetes_namespace.amazon_cloudwatch.metadata[0].name
    annotations = {
      "eks.amazonaws.com/role-arn" = aws_iam_role.fluent_bit.arn
    }
  }
}

# Fluent Bit ConfigMap
resource "kubernetes_config_map" "fluent_bit" {
  metadata {
    name      = "fluent-bit-config"
    namespace = kubernetes_namespace.amazon_cloudwatch.metadata[0].name
    labels = {
      app = "fluent-bit"
    }
  }

  data = {
    "fluent-bit.conf" = <<-EOF
      [SERVICE]
          Flush         5
          Log_Level     info
          Daemon        off
          Parsers_File  parsers.conf
          HTTP_Server   On
          HTTP_Listen   0.0.0.0
          HTTP_Port     2020

      @INCLUDE input-kubernetes.conf
      @INCLUDE filter-kubernetes.conf
      @INCLUDE output-cloudwatch.conf
    EOF

    "input-kubernetes.conf" = <<-EOF
      [INPUT]
          Name              tail
          Tag               kube.*
          Path              /var/log/containers/*.log
          Parser            docker
          DB                /fluent-bit/db/flb_kube.db
          Mem_Buf_Limit     50MB
          Skip_Long_Lines   On
          Refresh_Interval  10
    EOF

    "filter-kubernetes.conf" = <<-EOF
      [FILTER]
          Name                kubernetes
          Match               kube.*
          Kube_URL            https://kubernetes.default.svc:443
          Kube_CA_File        /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
          Kube_Token_File     /var/run/secrets/kubernetes.io/serviceaccount/token
          Kube_Tag_Prefix     kube.var.log.containers.
          Merge_Log           On
          Merge_Log_Key       log_processed
          K8S-Logging.Parser  On
          K8S-Logging.Exclude Off
    EOF

    "output-cloudwatch.conf" = <<-EOF
      [OUTPUT]
          Name                cloudwatch_logs
          Match               kube.*
          region              ${var.aws_region}
          log_group_name      /aws/eks/${var.project_name}-${var.environment}/containers
          log_stream_prefix   fluentbit-
          auto_create_group   true
    EOF

    "parsers.conf" = <<-EOF
      [PARSER]
          Name        docker
          Format      json
          Time_Key    time
          Time_Format %Y-%m-%dT%H:%M:%S.%L
          Time_Keep   On

      [PARSER]
          Name        syslog
          Format      regex
          Regex       ^<(?<pri>[0-9]+)>(?<time>[^ ]* {1,2}[^ ]* [^ ]*) (?<host>[^ ]*) (?<ident>[a-zA-Z0-9_\/\.\-]*)(?:\[(?<pid>[0-9]+)\])?(?:[^\:]*\:)? *(?<message>.*)$
          Time_Key    time
          Time_Format %b %d %H:%M:%S
    EOF
  }
}

# Fluent Bit ClusterRole
resource "kubernetes_cluster_role" "fluent_bit" {
  metadata {
    name = "fluent-bit"
  }

  rule {
    api_groups = [""]
    resources  = ["namespaces", "pods", "pods/logs"]
    verbs      = ["get", "list", "watch"]
  }
}

# Fluent Bit ClusterRoleBinding
resource "kubernetes_cluster_role_binding" "fluent_bit" {
  metadata {
    name = "fluent-bit"
  }

  role_ref {
    api_group = "rbac.authorization.k8s.io"
    kind      = "ClusterRole"
    name      = kubernetes_cluster_role.fluent_bit.metadata[0].name
  }

  subject {
    kind      = "ServiceAccount"
    name      = kubernetes_service_account.fluent_bit.metadata[0].name
    namespace = kubernetes_namespace.amazon_cloudwatch.metadata[0].name
  }
}

# Fluent Bit DaemonSet
resource "kubernetes_daemonset" "fluent_bit" {
  metadata {
    name      = "fluent-bit"
    namespace = kubernetes_namespace.amazon_cloudwatch.metadata[0].name
    labels = {
      app     = "fluent-bit"
      version = "v1"
    }
  }

  spec {
    selector {
      match_labels = {
        app = "fluent-bit"
      }
    }

    template {
      metadata {
        labels = {
          app     = "fluent-bit"
          version = "v1"
        }
      }

      spec {
        service_account_name = kubernetes_service_account.fluent_bit.metadata[0].name

        # Tolerate all taints to run on all nodes including GPU nodes
        toleration {
          operator = "Exists"
        }

        container {
          name  = "fluent-bit"
          image = "public.ecr.aws/aws-observability/aws-for-fluent-bit:stable"

          port {
            container_port = 2020
            name           = "http"
          }

          resources {
            requests = {
              cpu    = "100m"
              memory = "128Mi"
            }
            limits = {
              cpu    = "500m"
              memory = "256Mi"
            }
          }

          volume_mount {
            name       = "varlog"
            mount_path = "/var/log"
            read_only  = true
          }

          volume_mount {
            name       = "varlibdockercontainers"
            mount_path = "/var/lib/docker/containers"
            read_only  = true
          }

          volume_mount {
            name       = "fluent-bit-config"
            mount_path = "/fluent-bit/etc/"
          }

          volume_mount {
            name       = "fluent-bit-db"
            mount_path = "/fluent-bit/db"
          }
        }

        volume {
          name = "varlog"
          host_path {
            path = "/var/log"
          }
        }

        volume {
          name = "varlibdockercontainers"
          host_path {
            path = "/var/lib/docker/containers"
          }
        }

        volume {
          name = "fluent-bit-config"
          config_map {
            name = kubernetes_config_map.fluent_bit.metadata[0].name
          }
        }

        volume {
          name = "fluent-bit-db"
          empty_dir {}
        }

        termination_grace_period_seconds = 10
      }
    }
  }

  depends_on = [module.node_groups]
}

# --- KubeRay Operator ---
resource "helm_release" "kuberay_operator" {
  name             = "kuberay-operator"
  repository       = "https://ray-project.github.io/kuberay-helm/"
  chart            = "kuberay-operator"
  version          = "1.3.0"
  namespace        = "kuberay-system"
  create_namespace = true

  depends_on = [module.node_groups]
}
