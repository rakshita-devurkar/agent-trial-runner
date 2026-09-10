# agent-trial-runner

Runs an AI agent against the same task many times, in clean isolated
environments, and reports which version of the agent is actually better.

## Why many times

An agent asked the same question twice does not answer the same way twice. One
trial cannot tell you whether a pass was skill or luck, so every task is run
repeatedly and the result is a rate, not a verdict. Comparing rates is the only
comparison worth making.

## Why it grades records instead of replies

An agent can say *"Done, I've cancelled that for you"* while having cancelled
nothing, or the wrong thing. So a trial is graded by comparing the booking
records it left behind against what the task required. Nothing reads the
agent's own account of what it did.

## Status

Working: the travel domain, six tools, the grader, and one trial end to end
against Amazon Bedrock -- with budgets on steps, tokens and time, a full trace
for replay, and infrastructure errors separated from wrong answers so a
throttled request never counts as a failing agent.

Next: the runner that executes many trials at once, then the task generator.
