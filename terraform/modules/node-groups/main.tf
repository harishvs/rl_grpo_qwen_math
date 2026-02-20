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

# GPU Node Group EFA Security Group
resource "aws_security_group" "efa" {
  count = var.efa_enabled ? 1 : 0

  name        = "${local.name_prefix}-efa-sg"
  description = "Security group for EFA - allows all traffic between GPU nodes"
  vpc_id      = var.vpc_id

  ingress {
    from_port = 0
    to_port   = 0
    protocol  = "-1"
    self      = true
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.tags, {
    Name = "${local.name_prefix}-efa-sg"
  })
}

# Launch template with EFA for GPU nodes
resource "aws_launch_template" "gpu_efa" {
  count = var.efa_enabled ? 1 : 0

  name = "${local.name_prefix}-gpu-efa"

  instance_type = var.capacity_reservation_id != "" ? var.gpu_instance_types[0] : null

  dynamic "capacity_reservation_specification" {
    for_each = var.capacity_reservation_id != "" ? [1] : []
    content {
      capacity_reservation_preference = "capacity-reservations-only"
      capacity_reservation_target {
        capacity_reservation_id = var.capacity_reservation_id
      }
    }
  }

  dynamic "instance_market_options" {
    for_each = var.capacity_reservation_id != "" ? [1] : []
    content {
      market_type = "capacity-block"
    }
  }

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size = var.gpu_disk_size
      volume_type = "gp3"
    }
  }

  # Primary network interface - network card 0
  network_interfaces {
    device_index                = 0
    network_card_index          = 0
    interface_type              = "efa"
    security_groups             = [aws_security_group.efa[0].id, var.cluster_security_group_id]
    delete_on_termination       = true
  }

  # EFA on network card 1
  network_interfaces {
    device_index                = 1
    network_card_index          = 1
    interface_type              = "efa"
    security_groups             = [aws_security_group.efa[0].id, var.cluster_security_group_id]
    delete_on_termination       = true
  }

  # EFA on network card 2
  network_interfaces {
    device_index                = 2
    network_card_index          = 2
    interface_type              = "efa"
    security_groups             = [aws_security_group.efa[0].id, var.cluster_security_group_id]
    delete_on_termination       = true
  }

  # EFA on network card 3
  network_interfaces {
    device_index                = 3
    network_card_index          = 3
    interface_type              = "efa"
    security_groups             = [aws_security_group.efa[0].id, var.cluster_security_group_id]
    delete_on_termination       = true
  }

  tag_specifications {
    resource_type = "instance"
    tags = merge(var.tags, {
      Name = "${local.name_prefix}-gpu-efa-node"
    })
  }

  tags = var.tags
}

# GPU Node Group (g5.48xlarge for trainer/actor/reference)
resource "aws_eks_node_group" "gpu" {
  cluster_name    = var.cluster_name
  node_group_name = "${local.name_prefix}-gpu"
  node_role_arn   = aws_iam_role.node.arn
  subnet_ids      = length(var.gpu_subnet_ids) > 0 ? var.gpu_subnet_ids : var.subnet_ids
  instance_types  = var.capacity_reservation_id != "" ? null : var.gpu_instance_types
  ami_type        = "AL2023_x86_64_NVIDIA"
  capacity_type   = var.capacity_reservation_id != "" ? "CAPACITY_BLOCK" : null

  dynamic "launch_template" {
    for_each = var.efa_enabled ? [1] : []
    content {
      id      = aws_launch_template.gpu_efa[0].id
      version = aws_launch_template.gpu_efa[0].latest_version
    }
  }

  disk_size = var.efa_enabled ? null : var.gpu_disk_size

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
  ami_type        = "AL2023_x86_64_STANDARD"
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
