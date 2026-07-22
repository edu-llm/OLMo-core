# Copy to env.sh and `source` it before submitting jobs. Do NOT commit real secrets.
#
# S3 locations (created in us-east-2, shared sbsandbox account).
export OLMO_CHECKPOINT_S3="s3://alphaai-edullm-checkpoints"
export OLMO_DATA_S3="s3://alphaai-edullm-data"

# AWS region + credentials for boto3 on the compute node. On an EC2 instance with an instance
# role, leave S3_PROFILE unset (default chain picks up the role). Set S3_PROFILE only if using a
# named profile.
export AWS_DEFAULT_REGION="us-east-2"
# export S3_PROFILE="sbsandbox"

# Repo the scheduler pulls each job's branch from.
export OLMO_REPO_URL="https://github.com/edu-llm/OLMo-core.git"

# Cluster label passed to the training CLI (any non-"ai2/" string; keeps us off Beaker paths).
export OLMO_CLUSTER_LABEL="aws"

# Environment source on the compute node — set exactly one path/image.
export OLMO_VENV="$HOME/olmo-venv"                 # Option B: shared virtualenv
# export OLMO_ECR_IMAGE="<acct>.dkr.ecr.us-east-2.amazonaws.com/olmo-core:stable"  # Option A: container

# Experiment tracking (use your own WandB entity/project; keep the key out of git).
export WANDB_API_KEY="REPLACE"
