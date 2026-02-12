# Node Groups Module Outputs

output "gpu_node_group_arn" {
  description = "ARN of the GPU node group"
  value       = aws_eks_node_group.gpu.arn
}

output "gpu_node_group_id" {
  description = "ID of the GPU node group"
  value       = aws_eks_node_group.gpu.id
}

output "cpu_node_group_arn" {
  description = "ARN of the CPU node group"
  value       = aws_eks_node_group.cpu.arn
}

output "cpu_node_group_id" {
  description = "ID of the CPU node group"
  value       = aws_eks_node_group.cpu.id
}

output "node_role_arn" {
  description = "ARN of the node IAM role"
  value       = aws_iam_role.node.arn
}
