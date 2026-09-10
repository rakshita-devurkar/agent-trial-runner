# agent-trial-runner

Runs an AI agent against the same task many times, in clean isolated
environments, and reports which version of the agent is actually better.

Built for a matrix of **2,500 tasks × 12 configurations × 10 repetitions =
300,000 trials**, on Amazon Bedrock.

## The two ideas it is built on

**An agent asked the same question twice does not answer the same way twice.**
One trial cannot tell you whether a pass was skill or luck, so every task is run
repeatedly and the result is a rate. A configuration that passes a task 6 times
in 10 is not passing; it is a coin the user has to flip, and only repetition can
see that.

**An agent can say "Done, I've cancelled that" while having cancelled nothing.**
So a trial is graded by comparing the booking records it left behind against
what the task required. Nothing reads the agent's own account of what it did.

## One trial, end to end

```
1. Pull the next cell of the matrix
     { task: 1847, config: "micro/plan-first", repetition: 7 }

2. Copy the records for this trial only
     flights, hotels, bookings — nothing shared with any other trial

3. Load the pinned prompt
     "plan-first@9f2ac1" — content-addressed, cannot change silently

4. Agent loop, capped at 8 steps / 45 seconds / 20k tokens
     ask Bedrock → it calls a tool → record → repeat

5. Grade by comparing records to what the task required
     match = pass, mismatch = fail

6. Write the row to DynamoDB, the trace to S3
     traces only for trials that did not pass
```

Repeat 300,000 times, 48 at once.

## Architecture

```
  YOUR LAPTOP
  ┌────────────────────────────────────────────┐
  │  launch_ec2.py march-01                    │
  │  trial-runner --aws report march-01        │──── watch it, from anywhere
  └───────────────────┬────────────────────────┘
                      │ launch, with a startup script
                      ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  EC2  t4g.small                          IAM role attached  │
  │                                          (no keys on disk)  │
  │  boot → git clone → uv sync → run → shut itself down        │
  │                                                             │
  │   ┌──────────────┐   generate 2,500 tasks                   │
  │   │  the matrix  │   × 12 configs × 10 reps                 │
  │   │  300,000     │   minus what DynamoDB says is done       │
  │   └──────┬───────┘                                          │
  │          ▼                                                  │
  │   ┌──────────────┐                                          │
  │   │    queue     │                                          │
  │   └──────┬───────┘                                          │
  │          ▼                                                  │
  │   ┌────────────────────────────┐                            │
  │   │ 48 workers                 │                            │
  │   │   each running one trial ──┼───┐  ≤25 requests/sec      │
  │   │   at a time, as above      │   │  (self-adjusting)      │
  │   └────────────────────────────┘   │                        │
  └────────────────┬───────────────────┼────────────────────────┘
                   │                   │
      ┌────────────┴─────┐             ▼
      ▼                  ▼      ┌──────────────┐
 ┌──────────┐      ┌──────────┐ │   BEDROCK    │
 │ DYNAMODB │      │    S3    │ │              │
 │          │      │          │ │ nova-micro   │
 │ 1 row    │      │ traces,  │ │ nova-lite    │
 │ per      │      │ failures │ │ ministral-3b │
 │ trial    │      │ only     │ │ gpt-oss-20b  │
 └──────────┘      └──────────┘ └──────────────┘
  what you          what you      the models
  count             read          under test
```

Bedrock is a separate service, not something running on the instance -- the EC2
box calls it over the network with exactly the code that runs on a laptop. It is
also the only part that costs money; the instance, the table and the bucket all
sit inside the free tier at this volume.

Notably absent: Lambda and SQS. Every trial is waiting on Bedrock, and Bedrock's
rate limit belongs to the account rather than the machine, so more machines
cannot go faster. A single process coordinating its own rate is the better
design; split across Lambdas, each one would have to find that ceiling
separately by being refused.

Nobody logs in to start the run. The launcher hands AWS a startup script, and
the machine clones, installs, runs and stops itself. Progress is read from
DynamoDB rather than from the instance, which is also why a machine that dies
mid-run is replaceable: launch another with the same run name.

## Running it

```bash
uv sync

# locally, small
uv run trial-runner run smoke --tasks 5 --repetitions 3 --only micro

# on AWS, for real
uv run python infra/setup_aws.py           # bucket + table
uv run python infra/setup_ec2_role.py      # the instance's permissions
uv run python infra/launch_ec2.py march-01 # 300,000 trials
uv run trial-runner --aws report march-01  # from anywhere, while it runs
```

Re-running a run name **resumes** it rather than starting over. A five-hour run
that dies at hour four continues from hour four — on a different machine if
necessary, because the results and the resume state live in DynamoDB and not on
the instance.

## What the platform guarantees

Each of these is a way eval results go wrong quietly, which is the only way that
matters -- a run that crashes gets fixed, a run that reports a plausible wrong
number gets believed.

- **A trial that could not run is not a trial that failed.** Throttling,
  timeouts and crashes are recorded apart from wrong answers and kept out of
  every rate. Counted together, a pass rate is partly a measurement of our own
  outages, and it flatters whichever configuration happened to run while the
  service was healthy.
- **Resuming cannot double-count.** Each cell is written under a conditional
  put, so a re-attempted trial is refused by the database rather than counted
  twice. Without it a resumed run quietly stops being a rate.
- **The corpus is provably solvable.** An oracle that does exactly what each
  task asks, through the same tools an agent has, scores 100%. A generated task
  with an impossible expectation would fail every correct agent forever and look
  like a model weakness.
- **Every result records what produced it,** and every trial runs isolated and
  capped. Model, prompt version, corpus seed and grader version travel with each
  row; each trial gets its own copy of the records and its own budget of steps,
  tokens and seconds.

## What is being tested, and against what

Twelve configurations -- four Bedrock models times three prompt styles -- run
against 2,500 generated travel tasks. The models were probed rather than picked:
of seventeen candidates only thirteen accept tool calls at all. The tasks are
generated so that each one's correct answer is known by construction rather than
written down, and an oracle proves the whole corpus is solvable.

**[The configurations and the corpus in full](docs/corpus.md)**

## Not done yet

- **Trace replay.** Every trial's steps are recorded to S3, but there is no way
  to re-run one from its trace. Reading a failure back still means reading JSON
  rather than watching it happen, and reproducing one means paying the model
  again.
- **A release gate.** Nothing turns a set of rates into a ship/do-not-ship
  decision.
- **Statistical comparison.** Twelve configurations is sixty-six pairwise
  comparisons, and some will differ by chance alone. Raw pass rates are not safe
  to rank on.

## Development

```bash
uv sync --extra dev
uv run pytest              # 61 tests, none of which call Bedrock
uv run ruff check . && uv run mypy
```

Isolation, worker bounds, budgets, throttling, resuming and idempotence are all
tested against fakes. They are properties of the platform, not of any model, and
testing them against a live service would make them slow, costly and
non-deterministic for nothing.
