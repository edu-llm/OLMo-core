"""The three numbers that decide how a demonstration is run.

    python src/scripts/downstream_lane/measure_the_endpoint.py --endpoint http://127.0.0.1:8000/v1 --model edullm

Time to first token, tokens per second once it starts, and what all of that becomes when
three people type at the same time. Nothing else is measured, because nothing else
changes what somebody standing in front of a room decides to do.

TIME TO FIRST TOKEN IS THE ONE THAT DECIDES THE SCRIPT, AND IT IS NOT THROUGHPUT. A
presenter who knows the first token lands in under a second types the question and waits;
one whose endpoint takes eight seconds has to say something over the gap, and finding
that out live is how a demonstration turns into an apology. The two are measured
separately because a model can be fast at one and slow at the other, and prefix length
moves the first while barely touching the second -- so a long prompt is measured too.

CONCURRENCY IS MEASURED AS SLOWDOWN AND NOT AS AGGREGATE THROUGHPUT. Total tokens per
second across three streams goes up under batching and would read as good news, while
what each of the three people is looking at gets slower. The per-stream figure is the one
in the room.

Only the OpenAI-compatible endpoint is touched, so this runs against anything serving
that protocol -- including through the chat page's proxy, which is worth doing at least
once because the proxy is what the audience actually goes through.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

#: A prompt long enough to make prefill visible against the noise, and short enough that
#: no model in this family refuses it for length.
LONG_PROMPT = (
    "Here is a passage about the history of writing systems. " * 40
    + "\n\nSummarise the passage above in one sentence."
)


@dataclass
class Stream:
    first_token_seconds: float
    total_seconds: float
    tokens: int
    text: str

    @property
    def tokens_per_second(self) -> float:
        generating = self.total_seconds - self.first_token_seconds
        return self.tokens / generating if generating > 0 and self.tokens else 0.0


def one_stream(endpoint: str, model: str, prompt: str, *, max_tokens: int) -> Stream:
    """One streamed completion, timed from the moment the request leaves.

    Counted in server-sent events rather than by re-tokenising the reply, because the
    thing a reader perceives as speed is deltas arriving and vLLM emits one token per
    delta. Re-tokenising would measure the tokenizer as well as the server, and would
    disagree with the count for exactly the replies that matter.
    """
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
    ).encode()
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    started = time.perf_counter()
    first: float | None = None
    tokens = 0
    text: list[str] = []
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                piece = json.loads(payload)
            except ValueError:
                continue
            delta = (piece.get("choices") or [{}])[0].get("delta", {}).get("content")
            if not delta:
                continue
            if first is None:
                first = time.perf_counter() - started
            tokens += 1
            text.append(delta)
    finished = time.perf_counter() - started
    return Stream(first if first is not None else finished, finished, tokens, "".join(text))


def _report(label: str, streams: list[Stream]) -> None:
    firsts = [s.first_token_seconds for s in streams]
    rates = [s.tokens_per_second for s in streams]
    print(
        f"{label:<34} first token {statistics.median(firsts):5.2f}s "
        f"(worst {max(firsts):5.2f}s)   {statistics.median(rates):6.1f} tok/s per stream   "
        f"{sum(s.tokens for s in streams)} tokens over {len(streams)} stream(s)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="edullm")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3, help="how many times each single-stream case is run")
    parser.add_argument("--concurrency", type=int, default=3, help="how many people are typing at once")
    arguments = parser.parse_args()

    # The first request after a load pays for CUDA graph capture and a cold cache and is
    # not what anybody will see, so it is spent deliberately rather than averaged in. It is
    # also the honest number for "the very first question of the demonstration", which is
    # why it is printed instead of discarded.
    warm = one_stream(arguments.endpoint, arguments.model, "Hello.", max_tokens=16)
    print(f"{'first request after load':<34} first token {warm.first_token_seconds:5.2f}s")

    short = [
        one_stream(arguments.endpoint, arguments.model, "What is the capital of Japan?", max_tokens=arguments.max_tokens)
        for _ in range(arguments.repeats)
    ]
    _report("one person, short prompt", short)

    long_prompt = [
        one_stream(arguments.endpoint, arguments.model, LONG_PROMPT, max_tokens=arguments.max_tokens)
        for _ in range(arguments.repeats)
    ]
    _report("one person, ~500 token prompt", long_prompt)

    with ThreadPoolExecutor(max_workers=arguments.concurrency) as pool:
        together = list(
            pool.map(
                lambda n: one_stream(
                    arguments.endpoint,
                    arguments.model,
                    f"Question {n}: what is the capital of Japan?",
                    max_tokens=arguments.max_tokens,
                ),
                range(arguments.concurrency),
            )
        )
    _report(f"{arguments.concurrency} people at once", together)

    alone = statistics.median([s.tokens_per_second for s in short])
    crowded = statistics.median([s.tokens_per_second for s in together])
    if alone:
        print(f"\nthree at once costs each of them {100 * (1 - crowded / alone):.0f}% of their speed")
    print(f"\nsample reply:\n{short[0].text.strip()[:400]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
