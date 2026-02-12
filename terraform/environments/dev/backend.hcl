# Backend configuration for dev environment
# Usage: terraform init -backend-config=environments/dev/backend.hcl

# S3 bucket for storing Terraform state
# Note: This bucket must be created before running terraform init
bucket = "rl-code-llm-training-tfstate-dev"

# Path to the state file within the bucket
key = "dev/terraform.tfstate"

# AWS region for the S3 bucket and DynamoDB table
region = "us-east-1"

# DynamoDB table for state locking
# Note: This table must be created before running terraform init
# Table should have a primary key named "LockID" (String)
dynamodb_table = "rl-code-llm-training-tflock-dev"

# Enable server-side encryption for state file
encrypt = true
