#!/bin/bash
# Teardown EKS cluster and all associated resources.
# Run with --dry-run first to see what will be deleted.
#
# Usage:
#   ./scripts/infra/teardown-eks.sh --dry-run    # Preview only
#   ./scripts/infra/teardown-eks.sh               # Actually delete
set -euo pipefail

CLUSTER_NAME="${CLUSTER_NAME:-rl-code-llm-training-dev}"
REGION="${REGION:-us-east-1}"
DRY_RUN=false

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --cluster) CLUSTER_NAME="$2"; shift 2 ;;
        --region) REGION="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ "${DRY_RUN}" == "true" ]]; then
    echo "=== DRY RUN -- nothing will be deleted ==="
fi

run() {
    echo "  -> $*"
    if [[ "${DRY_RUN}" == "false" ]]; then
        eval "$@" || echo "  !! Failed (continuing...)"
    fi
}

echo ""
echo "=== Tearing down EKS cluster: ${CLUSTER_NAME} (${REGION}) ==="
echo ""

# --- 1. Delete K8s workloads (prevents finalizer hangs) ---
echo "--- Step 1: Delete K8s workloads ---"
if kubectl cluster-info > /dev/null 2>&1; then
    run "kubectl delete --all deployments --all-namespaces --ignore-not-found 2>/dev/null || true"
    run "kubectl delete --all statefulsets --all-namespaces --ignore-not-found 2>/dev/null || true"
    run "kubectl delete --all daemonsets -n default --ignore-not-found 2>/dev/null || true"
    run "kubectl delete --all jobs --all-namespaces --ignore-not-found 2>/dev/null || true"
    run "kubectl delete --all services --all-namespaces --ignore-not-found 2>/dev/null || true"
    # Delete Helm releases
    run "helm ls -A --short | xargs -r -L1 helm uninstall 2>/dev/null || true"
    echo "  Waiting 30s for resources to clean up..."
    [[ "${DRY_RUN}" == "false" ]] && sleep 30
else
    echo "  kubectl not connected, skipping workload cleanup"
fi

# --- 2. Delete EKS addons ---
echo ""
echo "--- Step 2: Delete EKS addons ---"
for ADDON in $(aws eks list-addons --cluster-name "${CLUSTER_NAME}" --region "${REGION}" --query 'addons[]' --output text 2>/dev/null || echo ""); do
    run "aws eks delete-addon --cluster-name ${CLUSTER_NAME} --addon-name ${ADDON} --region ${REGION}"
done

# --- 3. Delete node groups ---
echo ""
echo "--- Step 3: Delete node groups ---"
NODE_GROUPS="${CLUSTER_NAME}-cpu ${CLUSTER_NAME}-gpu"
for NG in ${NODE_GROUPS}; do
    echo "  Deleting node group: ${NG}"
    run "aws eks delete-nodegroup --cluster-name ${CLUSTER_NAME} --nodegroup-name ${NG} --region ${REGION}"
done

if [[ "${DRY_RUN}" == "false" ]]; then
    echo "  Waiting for node groups to delete (this takes 5-10 minutes)..."
    for NG in ${NODE_GROUPS}; do
        aws eks wait nodegroup-deleted --cluster-name "${CLUSTER_NAME}" --nodegroup-name "${NG}" --region "${REGION}" 2>/dev/null || true
    done
fi

# --- 4. Delete the EKS cluster ---
echo ""
echo "--- Step 4: Delete EKS cluster ---"
run "aws eks delete-cluster --name ${CLUSTER_NAME} --region ${REGION}"

if [[ "${DRY_RUN}" == "false" ]]; then
    echo "  Waiting for cluster to delete (this takes 5-10 minutes)..."
    aws eks wait cluster-deleted --name "${CLUSTER_NAME}" --region "${REGION}" 2>/dev/null || true
fi

# --- 5. Delete IAM roles ---
echo ""
echo "--- Step 5: Delete IAM roles ---"
IAM_ROLES=(
    "${CLUSTER_NAME}-cluster-role"
    "${CLUSTER_NAME}-node-role"
    "${CLUSTER_NAME}-grpo-trainer-role"
    "${CLUSTER_NAME}-ebs-csi-driver-role"
    "${CLUSTER_NAME}-fsx-csi-driver-role"
    "${CLUSTER_NAME}-fluent-bit-role"
)

for ROLE in "${IAM_ROLES[@]}"; do
    echo "  Deleting role: ${ROLE}"
    # Detach all managed policies
    for POLICY_ARN in $(aws iam list-attached-role-policies --role-name "${ROLE}" --query 'AttachedPolicies[*].PolicyArn' --output text 2>/dev/null || echo ""); do
        [[ -z "${POLICY_ARN}" || "${POLICY_ARN}" == "None" ]] && continue
        run "aws iam detach-role-policy --role-name ${ROLE} --policy-arn ${POLICY_ARN}"
    done
    # Delete inline policies
    for POLICY_NAME in $(aws iam list-role-policies --role-name "${ROLE}" --query 'PolicyNames[]' --output text 2>/dev/null || echo ""); do
        [[ -z "${POLICY_NAME}" || "${POLICY_NAME}" == "None" ]] && continue
        run "aws iam delete-role-policy --role-name ${ROLE} --policy-name ${POLICY_NAME}"
    done
    # Delete the role
    run "aws iam delete-role --role-name ${ROLE}"
done

# --- 6. Delete OIDC provider ---
echo ""
echo "--- Step 6: Delete OIDC provider ---"
OIDC_URL=$(aws eks describe-cluster --name "${CLUSTER_NAME}" --region "${REGION}" --query 'cluster.identity.oidc.issuer' --output text 2>/dev/null || echo "")
if [[ -n "${OIDC_URL}" && "${OIDC_URL}" != "None" ]]; then
    OIDC_ID=$(echo "${OIDC_URL}" | sed 's|https://||')
    OIDC_ARN=$(aws iam list-open-id-connect-providers --query "OpenIDConnectProviderList[?ends_with(Arn, '${OIDC_ID}')].Arn" --output text 2>/dev/null || echo "")
    if [[ -n "${OIDC_ARN}" && "${OIDC_ARN}" != "None" ]]; then
        run "aws iam delete-open-id-connect-provider --open-id-connect-provider-arn ${OIDC_ARN}"
    fi
fi

# --- 7. Delete VPC (subnets, IGW, route tables, security groups, NAT gateways) ---
echo ""
echo "--- Step 7: Delete VPC ---"
VPC_ID=$(aws eks describe-cluster --name "${CLUSTER_NAME}" --region "${REGION}" --query 'cluster.resourcesVpcConfig.vpcId' --output text 2>/dev/null || echo "")
if [[ -z "${VPC_ID}" || "${VPC_ID}" == "None" ]]; then
    # Cluster already deleted -- try to find VPC by tag
    VPC_ID=$(aws ec2 describe-vpcs --filters "Name=tag:Project,Values=rl-code-llm-training" --query 'Vpcs[0].VpcId' --output text --region "${REGION}" 2>/dev/null || echo "")
fi
if [[ -z "${VPC_ID}" || "${VPC_ID}" == "None" ]]; then
    echo "  No VPC found -- skipping"
else
    echo "  VPC: ${VPC_ID}"

# Delete NAT gateways first (they take time)
for NAT_ID in $(aws ec2 describe-nat-gateways --filter "Name=vpc-id,Values=${VPC_ID}" --query 'NatGateways[*].NatGatewayId' --output text --region "${REGION}" 2>/dev/null || echo ""); do
    [[ -z "${NAT_ID}" || "${NAT_ID}" == "None" ]] && continue
    run "aws ec2 delete-nat-gateway --nat-gateway-id ${NAT_ID} --region ${REGION}"
done

if [[ "${DRY_RUN}" == "false" ]]; then
    echo "  Waiting 60s for NAT gateways to delete..."
    sleep 60
fi

# Release Elastic IPs
for ALLOC_ID in $(aws ec2 describe-addresses --filters "Name=domain,Values=vpc" --query "Addresses[?AssociationId==null].AllocationId" --output text --region "${REGION}" 2>/dev/null || echo ""); do
    [[ -z "${ALLOC_ID}" || "${ALLOC_ID}" == "None" ]] && continue
    run "aws ec2 release-address --allocation-id ${ALLOC_ID} --region ${REGION}"
done

# Detach and delete internet gateway
for IGW_ID in $(aws ec2 describe-internet-gateways --filters "Name=attachment.vpc-id,Values=${VPC_ID}" --query 'InternetGateways[*].InternetGatewayId' --output text --region "${REGION}" 2>/dev/null || echo ""); do
    [[ -z "${IGW_ID}" || "${IGW_ID}" == "None" ]] && continue
    run "aws ec2 detach-internet-gateway --internet-gateway-id ${IGW_ID} --vpc-id ${VPC_ID} --region ${REGION}"
    run "aws ec2 delete-internet-gateway --internet-gateway-id ${IGW_ID} --region ${REGION}"
done

# Delete subnets
for SUBNET_ID in $(aws ec2 describe-subnets --filters "Name=vpc-id,Values=${VPC_ID}" --query 'Subnets[*].SubnetId' --output text --region "${REGION}" 2>/dev/null || echo ""); do
    [[ -z "${SUBNET_ID}" || "${SUBNET_ID}" == "None" ]] && continue
    run "aws ec2 delete-subnet --subnet-id ${SUBNET_ID} --region ${REGION}"
done

# Delete non-default route tables
for RT_ID in $(aws ec2 describe-route-tables --filters "Name=vpc-id,Values=${VPC_ID}" --query 'RouteTables[?Associations[0].Main!=`true`].RouteTableId' --output text --region "${REGION}" 2>/dev/null || echo ""); do
    [[ -z "${RT_ID}" || "${RT_ID}" == "None" ]] && continue
    run "aws ec2 delete-route-table --route-table-id ${RT_ID} --region ${REGION}"
done

# Delete non-default security groups
for SG_ID in $(aws ec2 describe-security-groups --filters "Name=vpc-id,Values=${VPC_ID}" --query "SecurityGroups[?GroupName!='default'].GroupId" --output text --region "${REGION}" 2>/dev/null || echo ""); do
    [[ -z "${SG_ID}" || "${SG_ID}" == "None" ]] && continue
    run "aws ec2 delete-security-group --group-id ${SG_ID} --region ${REGION}"
done

# Delete the VPC
run "aws ec2 delete-vpc --vpc-id ${VPC_ID} --region ${REGION}"
fi

# --- 8. Delete S3 bucket ---
echo ""
echo "--- Step 8: Delete S3 checkpoint bucket ---"
S3_BUCKET="${CLUSTER_NAME}-checkpoints"
if aws s3 ls "s3://${S3_BUCKET}" --region "${REGION}" > /dev/null 2>&1; then
    run "aws s3 rb s3://${S3_BUCKET} --force --region ${REGION}"
fi

# --- 9. Delete ECR repositories ---
echo ""
echo "--- Step 9: Delete ECR repositories ---"
for REPO in "${CLUSTER_NAME}/trainer" "${CLUSTER_NAME}/environment" "${CLUSTER_NAME}/monarch-grpo"; do
    if aws ecr describe-repositories --repository-names "${REPO}" --region "${REGION}" > /dev/null 2>&1; then
        run "aws ecr delete-repository --repository-name ${REPO} --force --region ${REGION}"
    fi
done

# --- CloudWatch log groups: KEPT ---
# Training logs in CloudWatch are preserved for debugging and auditing.
# Delete manually if needed:
#   aws logs delete-log-group --log-group-name /aws/eks/rl-code-llm-training-dev/containers
#   aws logs delete-log-group --log-group-name /aws/eks/rl-code-llm-training-dev/application

echo ""
if [[ "${DRY_RUN}" == "true" ]]; then
    echo "=== DRY RUN COMPLETE -- nothing was deleted ==="
    echo "Run without --dry-run to execute."
else
    echo "=== Teardown complete ==="
fi
