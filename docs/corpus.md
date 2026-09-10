# What is being tested, and against what

The twelve configurations under test, and the 2,500 tasks they run against.

## The twelve configurations

Four models times three prompts, so the comparison can separate "which model"
from "which way of asking" -- and catch where they interact, since a cheap model
may need the careful prompt while a stronger one does not.

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

Prompts are content-addressed (`careful@9f2ac1`), so editing one produces a new
version rather than changing what an old result meant.

The models were probed, not picked off a list: of seventeen candidates on
Bedrock only thirteen accept tool calls at all -- Llama and Jamba reject the
request, Gemma accepts it and then never calls a tool. These four are the cheap
end of what works.

## The task corpus

2,500 tasks are generated, not written. A template invents its own records, so
it already knows which hotel is cheapest and which booking is the hotel one, and
the expected outcome is computed from that rather than typed in — a generated
task cannot disagree with its own answer.

Five shapes: book the cheapest hotel, book the cheapest flight, cancel one thing
while keeping another, rebook something cheaper, and book a flight and hotel
together. Across twenty cities, eight travellers and randomised prices, with
prices sampled without replacement so "the cheapest" is never a tie.
