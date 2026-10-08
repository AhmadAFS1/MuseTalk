#!/usr/bin/env bash
# Create all three H3 poses and persist their production MuseTalk caches to S3.
set -euo pipefail

avatar_workflow_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
avatar_repo_dir="$(cd "$avatar_workflow_dir/../.." && pwd)"
avatar_workspace_dir="$(cd "$avatar_repo_dir/.." && pwd)"

source "$avatar_repo_dir/scripts/lib/musetalk_env_layers.sh"
mt_env_load_file "$avatar_repo_dir/.runtime/h3_avatar_s3.env" "avatar-s3"
export AVATAR_S3_ENABLED="${AVATAR_S3_ENABLED:-1}"
export AVATAR_S3_BUCKET="${AVATAR_S3_BUCKET:-lingua-musetalk-s3-storage}"
export AVATAR_S3_PREFIX="${AVATAR_S3_PREFIX:-avatars}"
export AVATAR_S3_REGION="${AVATAR_S3_REGION:-us-east-1}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-$AVATAR_S3_REGION}"

# Planning and help do not need AWS access or a GPU.
avatar_needs_aws=1
avatar_previous_arg=''
for avatar_arg in "$@"; do
    case "$avatar_arg" in
        --help|-h|--stage=plan) avatar_needs_aws=0 ;;
        plan) [[ "$avatar_previous_arg" != --stage ]] || avatar_needs_aws=0 ;;
    esac
    avatar_previous_arg="$avatar_arg"
done
if [[ $# == 0 ]]; then
    set -- --help
    avatar_needs_aws=0
fi
if (( avatar_needs_aws )); then
    avatar_aws="${AWS_CLI:-$HOME/.local/bin/aws}"
    [[ -x "$avatar_aws" ]] || avatar_aws=aws
    "$avatar_aws" s3api head-bucket --bucket "$AVATAR_S3_BUCKET" \
        --region "$AVATAR_S3_REGION" --no-cli-pager
fi

exec python3 "$avatar_workflow_dir/batch_three_pose.py" \
    --workspace "$avatar_workspace_dir" --stage all --local-api \
    --expected-bucket "$AVATAR_S3_BUCKET" "$@"
