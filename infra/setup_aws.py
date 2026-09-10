"""Create the S3 bucket and DynamoDB table this project writes to.

A script rather than a console click-through, so the resources can be described,
reviewed and recreated. Safe to run twice: everything checks before it creates.

The DynamoDB key is deliberate. Partitioning by `run_id` and sorting by `cell`
means resuming a run is one query for that run rather than a scan of every trial
ever recorded, and the sort key groups a task's repetitions together.

Billing is on-demand: a run is hours of writes followed by weeks of nothing,
which is exactly the shape provisioned capacity handles worst.
"""

from __future__ import annotations

import argparse
import sys

import boto3
from botocore.exceptions import ClientError

TABLE = "agent-trial-runner-results"
REGION = "us-east-1"


def bucket_name(account_id: str) -> str:
    """S3 bucket names are globally unique, so the account id keeps it ours."""
    return f"agent-trial-runner-{account_id}"


def ensure_bucket(name: str, region: str) -> None:
    s3 = boto3.client("s3", region_name=region)
    try:
        s3.head_bucket(Bucket=name)
        print(f"  bucket {name} already exists")
        return
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in {"404", "NoSuchBucket", "403"}:
            raise

    kwargs = (
        {}
        if region == "us-east-1"
        else {"CreateBucketConfiguration": {"LocationConstraint": region}}
    )
    s3.create_bucket(Bucket=name, **kwargs)
    # Traces are internal artefacts; nothing here should ever be public.
    s3.put_public_access_block(
        Bucket=name,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    s3.put_bucket_encryption(
        Bucket=name,
        ServerSideEncryptionConfiguration={
            "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
        },
    )
    print(f"  created bucket {name} (private, encrypted)")


def ensure_table(name: str, region: str) -> None:
    dynamodb = boto3.client("dynamodb", region_name=region)
    try:
        dynamodb.describe_table(TableName=name)
        print(f"  table {name} already exists")
        return
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ResourceNotFoundException":
            raise

    dynamodb.create_table(
        TableName=name,
        KeySchema=[
            {"AttributeName": "run_id", "KeyType": "HASH"},
            {"AttributeName": "cell", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "run_id", "AttributeType": "S"},
            {"AttributeName": "cell", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    print(f"  creating table {name} ...")
    dynamodb.get_waiter("table_exists").wait(TableName=name)
    print(f"  created table {name} (on-demand billing)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default=REGION)
    args = parser.parse_args()

    account = boto3.client("sts", region_name=args.region).get_caller_identity()["Account"]
    bucket = bucket_name(account)

    print(f"account {account} in {args.region}")
    ensure_bucket(bucket, args.region)
    ensure_table(TABLE, args.region)
    print(f"\nbucket: {bucket}\ntable:  {TABLE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
