# Node Groups Module - GPU and CPU node groups for EKS

locals {
  name_prefix = "${var.project_name}-${var.environment}"
}

# Node IAM Role (shared by both node groups)
resource "aws_iam_role" "node" {
  name = "${local.name_prefix}-node-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "node_worker_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.node.name
}

resource "aws_iam_role_policy_attachment" "node_cni_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.node.name
}

resource "aws_iam_role_policy_attachment" "node_ecr_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.node.name
}

# S3 access for checkpoints
resource "aws_iam_role_policy" "node_s3_access" {
  name = "${local.name_prefix}-node-s3-access"
  role = aws_iam_role.node.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ]
      Resource = [
        "arn:aws:s3:::${local.name_prefix}-*",
        "arn:aws:s3:::${local.name_prefix}-*/*"
      ]
    }]
  })
}

# GPU Node Group (g5.48xlarge for trainer/actor/reference)
resource "aws_eks_node_group" "gpu" {
  cluster_name    = var.cluster_name
  node_group_name = "${local.name_prefix}-gpu"
  node_role_arn   = aws_iam_role.node.arn
  subnet_ids      = var.subnet_ids
  instance_types  = var.gpu_instance_types
  ami_type        = "AL2_x86_64_GPU"
  disk_size       = var.gpu_disk_size

  scaling_config {
    desired_size = var.gpu_desired_size
    min_size     = var.gpu_min_size
    max_size     = var.gpu_max_size
  }

  labels = {
    "node-type" = "gpu"
  }

  taint {
    key    = "nvidia.com/gpu"
    value  = "true"
    effect = "NO_SCHEDULE"
  }

  tags = merge(var.tags, {
    Name = "${local.name_prefix}-gpu-node"
  })

  depends_on = [
    aws_iam_role_policy_attachment.node_worker_policy,
    aws_iam_role_policy_attachment.node_cni_policy,
    aws_iam_role_policy_attachment.node_ecr_policy,
  ]
}

# CPU Node Group (c7i.large for environment/reward workers)
resource "aws_eks_node_group" "cpu" {
  cluster_name    = var.cluster_name
  node_group_name = "${local.name_prefix}-cpu"
  node_role_arn   = aws_iam_role.node.arn
  subnet_ids      = var.subnet_ids
  instance_types  = var.cpu_instance_types
  ami_type        = "AL2_x86_64"
  disk_size       = var.cpu_disk_size

  scaling_config {
    desired_size = var.cpu_desired_size
    min_size     = var.cpu_min_size
    max_size     = var.cpu_max_size
  }

  labels = {
    "node-type" = "cpu"
  }

  tags = merge(var.tags, {
    Name = "${local.name_prefix}-cpu-node"
  })

  depends_on = [
    aws_iam_role_policy_attachment.node_worker_policy,
    aws_iam_role_policy_attachment.node_cni_policy,
    aws_iam_role_policy_attachment.node_ecr_policy,
  ]
}
