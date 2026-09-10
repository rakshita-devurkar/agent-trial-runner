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

## What runs where

| Piece | Why |
|---|---|
| **Bedrock** | The models under test. The only thing that costs money |
| **EC2** (`t4g.small`) | Runs the matrix. Stops itself when finished |
| **DynamoDB** | One row per trial. Written as they happen, so a crash loses nothing |
| **S3** | Traces, for reading a failure back instead of reproducing it |
| **IAM role** | The instance's permissions. No credentials on the box |

Notably **not** used: Lambda and SQS. Every trial is waiting on Bedrock, and
Bedrock's rate limit belongs to the account rather than the machine — so more
machines cannot go faster, and a single process coordinating its own rate is
the better design. Splitting across Lambdas would mean discovering that ceiling
independently in each one.

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

- **Every trial starts from its own copy of the records.** One trial's booking
  can never become the next one's starting state.
- **A trial that could not run is not a trial that failed.** Throttling,
  timeouts and crashes are recorded apart from wrong answers and excluded from
  every rate — otherwise a pass rate is partly a measurement of our own outages.
- **Budgets are enforced per trial.** Steps, tokens and wall-clock. An agent
  that loops forever is a daily event at this volume, and running out is its own
  outcome rather than a silent failure.
- **Every result records what produced it.** Model, prompt version, corpus seed,
  grader version. A rate with no provenance cannot be compared with anything.
- **Resuming cannot double-count.** A cell is written under a conditional put,
  so re-attempting one is refused by the database rather than counted twice.
- **The corpus is provably solvable.** An oracle that does exactly what each
  task asks, through the same tools an agent has, scores 100%. If it ever
  doesn't, the corpus is broken rather than the model.

## The task corpus

2,500 tasks are generated, not written. A template invents its own records, so
it already knows which hotel is cheapest and which booking is the hotel one, and
the expected outcome is computed from that rather than typed in — a generated
task cannot disagree with its own answer.

Five shapes: book the cheapest hotel, book the cheapest flight, cancel one thing
while keeping another, rebook something cheaper, and book a flight and hotel
together. Across twenty cities, eight travellers and randomised prices, with
prices sampled without replacement so "the cheapest" is never a tie.

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
