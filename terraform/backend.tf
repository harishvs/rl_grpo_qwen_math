# Terraform Backend Configuration
# This file configures the S3 backend for remote state storage with DynamoDB locking.
# The actual backend configuration values are provided via backend.hcl files per environment.

terraform {
  backend "s3" {
    # Backend configuration is provided via -backend-config flag during terraform init
    # Example: terraform init -backend-config=environments/dev/backend.hcl
    #
    # Required backend.hcl values:
    # - bucket: S3 bucket name for state storage
    # - key: Path to state file within the bucket
    # - region: AWS region for the S3 bucket
    # - dynamodb_table: DynamoDB table name for state locking
    # - encrypt: Enable server-side encryption (should be true)
  }
}
