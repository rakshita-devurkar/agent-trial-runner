# Tripwire

An evaluation platform for AI travel agents.

Runs a large matrix of trials — many tasks, many agent configurations, many
repetitions each — in clean sandboxed environments, and reports which
configuration is actually better rather than which one got lucky.

## Why repetitions

An agent asked the same question twice does not give the same answer twice. A
single trial cannot tell you whether a pass was skill or luck. Repeating each
task turns a pass/fail into a rate, which is the only thing worth comparing.

## Status

Day one. Building the smallest possible slice first: one trial, running on
Lambda, calling Bedrock. Queue, storage and aggregation come after.
