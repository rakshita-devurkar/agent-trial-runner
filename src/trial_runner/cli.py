"""The command line: start a run, resume one, report on it.

Resuming is the same command. A run is named, and re-running the same name picks
up where it stopped rather than starting again -- because the failure mode this
guards against is a five-hour run dying at hour four, and at that point nobody
wants to think about which flag means "continue".
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from trial_runner.agent import Budget, Outcome
from trial_runner.generate import generate
from trial_runner.report import render, summarize, throughput
from trial_runner.runner import Config, run_matrix
from trial_runner.store import Store, manifest_for
from trial_runner.throttle import Limiter, Throttling, wrap_client
from trial_runner.versions import PROMPTS

#: The twelve configurations: four tool-capable models that differ in size and
#: vendor, times three prompts that differ along one axis each.
MODELS = [
    ("micro", "amazon.nova-micro-v1:0"),
    ("lite", "amazon.nova-lite-v1:0"),
    ("ministral", "mistral.ministral-3-3b-instruct"),
    ("gptoss", "openai.gpt-oss-20b-1:0"),
]


def configurations() -> list[Config]:
    return [
        Config(config_id=f"{short}/{prompt.name}", model_id=model_id, prompt=prompt)
        for short, model_id in MODELS
        for prompt in PROMPTS.values()
    ]


def build_store(args: argparse.Namespace) -> Any:
    """Local files while developing; DynamoDB and S3 for a real run."""
    if not args.aws:
        return Store(Path(args.out), args.run)

    import boto3

    from trial_runner.aws_store import AwsStore

    account = boto3.client("sts").get_caller_identity()["Account"]
    bucket_name = f"agent-trial-runner-{account}"
    return AwsStore(
        args.run,
        table=boto3.resource("dynamodb", region_name=args.region).Table(
            "agent-trial-runner-results"
        ),
        bucket=boto3.resource("s3", region_name=args.region).Bucket(bucket_name),
        bucket_name=bucket_name,
    )


def make_client_factory(args: argparse.Namespace, limiter: Limiter, stats: Throttling) -> Any:
    import boto3
    from botocore.config import Config as BotoConfig

    def factory() -> Any:
        client = boto3.client(
            "bedrock-runtime",
            region_name=args.region,
            # botocore's own retries handle transient network faults. Throttling
            # is handled a level up, where the shared limiter can also slow down.
            config=BotoConfig(retries={"max_attempts": 2, "mode": "standard"}, read_timeout=90),
        )
        return wrap_client(client, limiter=limiter, stats=stats)

    return factory


def cmd_run(args: argparse.Namespace) -> int:
    tasks = generate(args.tasks, seed=args.seed)
    configs = configurations()
    if args.only:
        configs = [c for c in configs if args.only in c.config_id]
    budget = Budget(max_steps=args.max_steps, max_seconds=args.max_seconds)

    store = build_store(args)
    already = store.done()
    total = len(tasks) * len(configs) * args.repetitions
    store.write_manifest(manifest_for(configs, args.seed, args.tasks, budget))

    print(f"run {args.run}")
    print(f"  {len(tasks)} tasks x {len(configs)} configs x {args.repetitions} reps = {total}")
    if already:
        print(f"  {len(already)} already done, resuming")
    print(f"  {args.workers} workers, {args.rate}/s")

    limiter = Limiter(rate=args.rate)
    stats = Throttling()
    started = time.monotonic()
    seen = {"n": 0, "pass": 0, "fail": 0, "other": 0}

    def on_result(result: Any) -> None:
        store.record(result)
        seen["n"] += 1
        if result.outcome is Outcome.PASS:
            seen["pass"] += 1
        elif result.outcome is Outcome.FAIL:
            seen["fail"] += 1
        else:
            seen["other"] += 1
        if seen["n"] % 25 == 0 or seen["n"] == total - len(already):
            elapsed = time.monotonic() - started
            rate = seen["n"] / elapsed if elapsed else 0
            remaining = (total - len(already) - seen["n"]) / rate if rate else 0
            print(
                f"\r  {seen['n']}/{total - len(already)}  "
                f"{seen['pass']}P {seen['fail']}F {seen['other']}? "
                f"{rate:.1f}/s  {remaining / 60:.0f}m left  "
                f"limit {limiter.rate:.0f}/s  {stats.refusals} throttled",
                end="",
                flush=True,
            )

    run_matrix(
        make_client_factory(args, limiter, stats),
        tasks,
        configs,
        repetitions=args.repetitions,
        workers=args.workers,
        budget=budget,
        on_result=on_result,
        skip=already,
    )
    store.close()
    elapsed = time.monotonic() - started
    done_now = seen["n"]
    print(f"\n\ndone in {elapsed / 60:.1f}m")
    if done_now:
        print(
            f"  {done_now / elapsed:.2f} trials/s sustained  |  "
            f"limiter settled at {limiter.rate:.0f}/s  |  "
            f"{stats.refusals} throttled, {stats.gave_up} abandoned, "
            f"{stats.waited_seconds:.0f}s spent waiting"
        )
        remaining = 300_000
        print(
            f"  at this rate 300,000 trials would take {remaining / (done_now / elapsed) / 3600:.1f}h"
        )
    print(render(summarize(store.results())))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    store = build_store(args)
    results = store.results()
    if not results:
        print(f"no results for run {args.run}")
        return 1
    summaries = summarize(results)
    print(render(summaries))
    print(f"\n{throughput(results)}")
    for config_id in sorted(summaries):
        summary = summaries[config_id]
        flaky = summary.flaky_tasks
        if flaky:
            print(f"\n{config_id}: {len(flaky)} flaky task(s), e.g. {flaky[:3]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trial-runner", description=__doc__)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--aws", action="store_true", help="Store in DynamoDB and S3.")
    parser.add_argument("--out", default=".runs", help="Where local runs are stored.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Start or resume a run.")
    run.add_argument("run", help="Run name. Re-using one resumes it.")
    run.add_argument("--tasks", type=int, default=25)
    run.add_argument("--repetitions", type=int, default=10)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--workers", type=int, default=16)
    run.add_argument("--rate", type=float, default=20.0, help="Requests per second to aim for.")
    run.add_argument("--max-steps", type=int, default=8)
    run.add_argument("--max-seconds", type=float, default=45.0)
    run.add_argument("--only", help="Only configs whose id contains this.")
    run.set_defaults(func=cmd_run)

    report = sub.add_parser("report", help="Summarize a finished run.")
    report.add_argument("run")
    report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
