"""Launch the instance that runs the trials, and stop it when it is done.

Three choices worth knowing about.

*No SSH key and no open ports.* The instance is reached through SSM Session
Manager instead, so nothing is listening on the internet and there is no private
key to lose. An eval runner has no reason to be reachable from outside AWS.

*It stops itself.* The most expensive mistake available here is an instance left
running for a month after a five-hour job. The run is wrapped so the machine
shuts down whether it succeeded or failed, and `InstanceInitiatedShutdownBehavior`
is set to stop rather than terminate so the logs survive for reading.

*The run is named on the command line.* Because results and resume state live in
DynamoDB rather than on the instance, a machine that dies mid-run is replaceable:
launch another with the same run name and it continues.
"""

from __future__ import annotations

import argparse
import sys

import boto3

PROFILE = "agent-trial-runner-instance"
REPO = "https://github.com/rakshita-devurkar/agent-trial-runner.git"

USER_DATA = """#!/bin/bash
set -xuo pipefail
exec > >(tee /var/log/trial-runner.log) 2>&1

echo "=== setup $(date -Is) ==="
dnf install -y git tar gzip
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

cd /opt
git clone --depth 1 {repo} runner
cd runner
uv sync --extra dev

echo "=== run $(date -Is) ==="
# `|| true`: the shutdown below must happen whether the run succeeds or not.
# An instance left running after a failure is the expensive failure mode here.
uv run trial-runner --aws run {run} \\
  --tasks {tasks} --repetitions {reps} \\
  --workers {workers} --rate {rate} || true

echo "=== done $(date -Is), stopping ==="
shutdown -h now
"""


def latest_al2023(region: str) -> str:
    """Resolved from SSM rather than hard-coded: AMI ids differ per region and
    change with every release."""
    ssm = boto3.client("ssm", region_name=region)
    image_id: str = ssm.get_parameter(
        Name="/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64"
    )["Parameter"]["Value"]
    return image_id


def launch(args: argparse.Namespace) -> str:
    ec2 = boto3.client("ec2", region_name=args.region)
    image_id = latest_al2023(args.region)
    user_data = USER_DATA.format(
        repo=REPO,
        run=args.run,
        tasks=args.tasks,
        reps=args.repetitions,
        workers=args.workers,
        rate=args.rate,
    )

    response = ec2.run_instances(
        ImageId=image_id,
        InstanceType=args.instance_type,
        MinCount=1,
        MaxCount=1,
        IamInstanceProfile={"Name": PROFILE},
        UserData=user_data,
        # Stop, not terminate: the machine is finished but its logs are not.
        InstanceInitiatedShutdownBehavior="stop",
        MetadataOptions={"HttpTokens": "required"},
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": f"trial-runner-{args.run}"},
                    {"Key": "Project", "Value": "agent-trial-runner"},
                    {"Key": "Run", "Value": args.run},
                ],
            }
        ],
    )
    instance_id: str = response["Instances"][0]["InstanceId"]
    return instance_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="Run name. Re-use it to resume on a new machine.")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--instance-type", default="t4g.small")
    parser.add_argument("--tasks", type=int, default=2500)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--rate", type=float, default=25.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    total = args.tasks * 12 * args.repetitions
    print(f"run {args.run}: {args.tasks} tasks x 12 configs x {args.repetitions} reps = {total:,}")
    if args.dry_run:
        print(f"would launch {args.instance_type} in {args.region}")
        return 0

    instance_id = launch(args)
    print(f"launched {instance_id} ({args.instance_type})")
    print("\nwatch it:")
    print(f"  aws ssm start-session --target {instance_id} --region {args.region}")
    print("  sudo tail -f /var/log/trial-runner.log")
    print("\nprogress, from anywhere:")
    print(f"  uv run trial-runner --aws report {args.run}")
    print("\nit stops itself when finished. to stop it early:")
    print(f"  aws ec2 stop-instances --instance-ids {instance_id} --region {args.region}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
