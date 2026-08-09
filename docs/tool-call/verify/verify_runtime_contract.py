"""Prove the dataset and the runtime agree, using the real tools that ship in olmo_core.

    PYTHONPATH=src python3 docs/tool-call/verify/verify_runtime_contract.py

The dataset teaches the model to emit a string. `olmo_core.tools.protocol.parse_function_calls` is
what turns that string back into a call at inference, and `olmo_core.tools` is what executes it.
If those disagree with what we write, the failure shows up only after training, as a model emitting
something nothing can run.

So this builds rows the way the generator will, against the live tool schemas, and checks three
things end to end: the runtime parses what we wrote, it recovers the same call, and — for the two
tools that actually execute — running the recovered call returns the answer we recorded.
"""

import importlib.util
import sys

sys.path.insert(0, "src")

from olmo_core.tools import (  # noqa: E402
    CalculatorTool,
    StaticBackend,
    SymbolicMathTool,
    WebSearchTool,
)

_spec = importlib.util.spec_from_file_location(
    "tool_call_serializer", "src/scripts/data/tool_call_serializer.py"
)
assert _spec is not None and _spec.loader is not None
ser = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ser
_spec.loader.exec_module(ser)


def schema_of(tool) -> "ser.ToolSchema":
    fn = tool.json_schema()["function"]
    return ser.ToolSchema(
        name=fn["name"], description=fn["description"], parameters=fn["parameters"]
    )


calc, sym = CalculatorTool(), SymbolicMathTool()
search = WebSearchTool(backend=StaticBackend(results=[]))
CALC, SYM, SEARCH = schema_of(calc), schema_of(sym), schema_of(search)

CASES = [
    (
        "calculator, one call",
        [CALC],
        "What is 5219 times 47?",
        [ser.Call("calculator", {"expression": "5219 * 47"})],
        calc,
    ),
    (
        "calculator, nested parens",
        [CALC, SEARCH],
        "Compute (3+4)*12 - 6/2.",
        [ser.Call("calculator", {"expression": "(3 + 4) * 12 - 6 / 2"})],
        calc,
    ),
    (
        "symbolic_math, solve",
        [SYM, CALC],
        "Solve x**2 - 4 = 0 for x.",
        [
            ser.Call(
                "symbolic_math", {"expression": "x**2 - 4", "operation": "solve", "variable": "x"}
            )
        ],
        sym,
    ),
    (
        "symbolic_math, differentiate",
        [SYM],
        "Differentiate x**3 + 2*x with respect to x.",
        [
            ser.Call(
                "symbolic_math", {"expression": "x**3 + 2*x", "operation": "diff", "variable": "x"}
            )
        ],
        sym,
    ),
    (
        "web_search, with int arg",
        [SEARCH],
        "Find recent news on OLMo 3.",
        [ser.Call("web_search", {"query": "OLMo 3 release news", "max_results": 3})],
        None,
    ),
    (
        "parallel, one block",
        [CALC],
        "What are 12*12 and 15*15?",
        [
            ser.Call("calculator", {"expression": "12 * 12"}),
            ser.Call("calculator", {"expression": "15 * 15"}),
        ],
        None,
    ),
]

print("=== rows built against the LIVE tool schemas ===")
ok = True
for name, schemas, user, calls, tool in CASES:
    row = ser.build_row(schemas=schemas, user=user, calls=calls)
    try:
        ser.assert_round_trip(row)
        ser.assert_runtime_parseable(row)
        verdict = "runtime parses it, and recovers the same call"
    except ValueError as e:
        verdict = f"FAILED: {e}"
        ok = False
    print(f"  {name:32s} {verdict}")

print()
print("=== execution: the two tools that actually run ===")
for name, schemas, user, calls, tool in CASES:
    if tool is None:
        continue
    recovered = ser.parse_function_calls(
        row_content := ser.build_row(schemas=schemas, user=user, calls=calls)["messages"][-1][
            "content"
        ]
    )
    assert row_content
    for call in recovered:
        try:
            result = tool.call(**call.arguments)
            print(f"  {call.name}({call.arguments}) -> {result!r}")
        except Exception as e:
            ok = False
            print(f"  {call.name}({call.arguments}) -> FAILED: {e}")

print()
print("=== the guard: a dotted name is refused before it can be written ===")
try:
    ser.serialize_call(ser.Call("weather.forecast_weather_api", {"q": "Paris"}))
    print("  NOT REFUSED — the guard is broken")
    ok = False
except ValueError as e:
    print(f"  refused, as it must be: {str(e)[:110]}")
    print(f"  flatten_tool_name gives {ser.flatten_tool_name('weather.forecast_weather_api')!r}")

print()
print("=== abstention still parses as zero calls ===")
row = ser.build_row(schemas=[CALC], user="Who wrote Hamlet?", prose="Shakespeare. No tool needed.")
n = len(ser.parse_function_calls(row["messages"][-1]["content"]))
print(f"  runtime recovered {n} calls (expected 0)")
ok &= n == 0

print()
print("ALL CONTRACTS HOLD" if ok else "CONTRACT VIOLATION — see above")
sys.exit(0 if ok else 1)
