# Infrastructure Rules

## Terraform-First Approach

All infrastructure changes MUST be done through Terraform to ensure:
- Repeatability across environments
- Version control of infrastructure state
- Proper dependency management
- Easy rollback capabilities

### What counts as infrastructure:
- AWS resources (EKS, S3, IAM, ECR, VPC, etc.)
- Kubernetes cluster-level resources (addons, CSI drivers, RBAC)
- Persistent storage (EBS volumes, EFS, FSx)
- Networking (load balancers, security groups, ingress)

### Process:
1. Make changes in `terraform/` directory
2. Run `terraform plan` to review changes
3. Run `terraform apply` to apply changes
4. Commit the Terraform code changes

### Do NOT:
- Use `aws` CLI to create/modify resources directly
- Use `kubectl` to create cluster-level resources that should be managed
- Make manual changes in AWS Console
- Create resources outside of Terraform that need to persist

### Exceptions:
- Debugging/troubleshooting (temporary resources)
- One-time migrations with proper documentation
- Application-level Kubernetes resources (deployments, services) that are managed separately
