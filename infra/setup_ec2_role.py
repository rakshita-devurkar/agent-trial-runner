"""An IAM role the runner instance assumes, so no credentials live on the box.

An EC2 instance with a role is handed short-lived credentials by AWS and they
rotate on their own. The alternative -- copying an access key onto the server --
puts a long-lived secret on a machine that may outlive the reason it existed,
and there is no way to tell later whether it leaked.

The policy is written narrow on purpose. This instance reads and writes exactly
one bucket and one table, and invokes models. It cannot create infrastructure,
read other buckets, or delete the results it just wrote. If it is ever
compromised, that list is the whole blast radius.
"""

from __future__ import annotations

import argparse
import json
import sys

import boto3
from botocore.exceptions import ClientError

ROLE = "agent-trial-runner-instance"
PROFILE = "agent-trial-runner-instance"
TABLE = "agent-trial-runner-results"

TRUST = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }
    ],
}


def policy(account: str, region: str, bucket: str) -> dict[str, object]:
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "InvokeModels",
                "Effect": "Allow",
                "Action": ["bedrock:InvokeModel", "bedrock:Converse"],
                "Resource": "*",
            },
            {
                "Sid": "ResultsTableOnly",
                "Effect": "Allow",
                "Action": [
                    "dynamodb:PutItem",
                    "dynamodb:Query",
                    "dynamodb:GetItem",
                    "dynamodb:BatchWriteItem",
                ],
                "Resource": f"arn:aws:dynamodb:{region}:{account}:table/{TABLE}",
            },
            {
                "Sid": "TracesBucketOnly",
                "Effect": "Allow",
                "Action": ["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
                "Resource": [f"arn:aws:s3:::{bucket}", f"arn:aws:s3:::{bucket}/*"],
            },
        ],
    }


def ensure_role(iam: object, account: str, region: str, bucket: str) -> None:
    client = boto3.client("iam")
    try:
        client.get_role(RoleName=ROLE)
        print(f"  role {ROLE} already exists")
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "NoSuchEntity":
            raise
        client.create_role(
            RoleName=ROLE,
            AssumeRolePolicyDocument=json.dumps(TRUST),
            Description="Runs agent trials: invoke models, write results and traces.",
        )
        print(f"  created role {ROLE}")

    # Put rather than create: re-running narrows the policy if it has changed.
    client.put_role_policy(
        RoleName=ROLE,
        PolicyName="runner-access",
        PolicyDocument=json.dumps(policy(account, region, bucket)),
    )
    print("  policy attached (one bucket, one table, model invocation)")

    try:
        client.get_instance_profile(InstanceProfileName=PROFILE)
        print(f"  instance profile {PROFILE} already exists")
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "NoSuchEntity":
            raise
        client.create_instance_profile(InstanceProfileName=PROFILE)
        client.add_role_to_instance_profile(InstanceProfileName=PROFILE, RoleName=ROLE)
        print(f"  created instance profile {PROFILE}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()

    account = boto3.client("sts").get_caller_identity()["Account"]
    bucket = f"agent-trial-runner-{account}"
    print(f"account {account}")
    ensure_role(None, account, args.region, bucket)
    print(f"\nattach to an instance with: --iam-instance-profile Name={PROFILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
