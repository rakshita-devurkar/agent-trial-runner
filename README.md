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
  │   │   copy the records         │                            │
  │   │   load the pinned prompt   │                            │
  │   │   run the agent loop  ─────┼───┐  ≤25 requests/sec      │
  │   │   grade against the records│   │  (self-adjusting)      │
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

Bedrock is a separate service, not something running on the instance. The EC2
box calls it over the network with exactly the code that runs on a laptop.

Nobody logs in to start the run. The launcher hands AWS a startup script, and
the machine clones, installs, runs and stops itself. Progress is read from
DynamoDB rather than from the instance, which is also why a machine that dies
mid-run is replaceable: launch another with the same run name.

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

## The twelve configurations

Four models times three prompts. Both axes vary so the comparison can separate
"which model" from "which way of asking", and show where they interact -- a
cheap model may need the careful prompt while a stronger one does not.

| Model | Why it is in the set |
| --- | --- |
| `amazon.nova-micro-v1:0` | Cheapest available, and the weakest. The floor |
| `amazon.nova-lite-v1:0` | Same family, one size up. Isolates size from vendor |
| `mistral.ministral-3-3b-instruct` | Tiny, different vendor entirely |
| `openai.gpt-oss-20b-1:0` | Open-weight, different family again |

| Prompt | What it says |
| --- | --- |
| `terse` | One line. Use the tools, do the request |
| `careful` | Adds the cost of getting it wrong: extra bookings are money, extra cancellations are lost trips |
| `plan-first` | Adds "look things up before acting, so you choose between real options rather than guessing at identifiers" |

Prompts are content-addressed (`careful@9f2ac1`). Editing one produces a new
version rather than changing what an old result meant.

Models were not picked from a list -- they were probed. Of seventeen candidates
on Bedrock, thirteen accept tool calls at all; Llama and Jamba reject the
request outright and Gemma accepts it and then never calls a tool. The four
above are the cheap end of what actually works.

## The task corpus

2,500 tasks are generated, not written. A template invents its own records, so
it already knows which hotel is cheapest and which booking is the hotel one, and
the expected outcome is computed from that rather than typed in — a generated
task cannot disagree with its own answer.

Five shapes: book the cheapest hotel, book the cheapest flight, cancel one thing
while keeping another, rebook something cheaper, and book a flight and hotel
together. Across twenty cities, eight travellers and randomised prices, with
prices sampled without replacement so "the cheapest" is never a tie.

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
