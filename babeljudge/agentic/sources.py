"""Synthetic task sources for agentic evaluation.

Two modes:
  synthetic_tool_tasks()  -- zero-network, self-contained demo tasks. Use this
                             for CI, quick iteration, and getting-started guides.
  from_bfcl()             -- Berkeley Function Calling Leaderboard dataset
                             (requires `datasets` package and network access).
"""
from __future__ import annotations

from typing import Optional

from .schema import AgentTrace, ToolCall, TraceStep


# ── synthetic tasks (zero-network) ───────────────────────────────────────────

_SYNTHETIC_TASKS = [
    {
        "task_id": "weather_query",
        "task": "What is the current weather in Paris and should I bring an umbrella?",
        "steps": [
            TraceStep(
                thought="I need to get the current weather for Paris.",
                tool_call=ToolCall("get_weather", {"city": "Paris", "units": "metric"}, "15°C, partly cloudy, 30% rain chance"),
            ),
            TraceStep(
                thought="Based on 30% rain chance, an umbrella is advisable.",
                tool_call=None,
                is_final=True,
            ),
        ],
        "answer": "Current weather in Paris: 15°C, partly cloudy. There is a 30% chance of rain, so I recommend bringing an umbrella.",
    },
    {
        "task_id": "stock_price",
        "task": "What is the current price of Apple stock and how has it changed this week?",
        "steps": [
            TraceStep(
                thought="I'll look up the current Apple stock price.",
                tool_call=ToolCall("get_stock_price", {"ticker": "AAPL"}, '{"price": 189.50, "currency": "USD"}'),
            ),
            TraceStep(
                thought="Now I'll get the weekly performance data.",
                tool_call=ToolCall("get_stock_history", {"ticker": "AAPL", "period": "1w"}, '{"open": 182.10, "close": 189.50, "change_pct": 4.06}'),
            ),
            TraceStep(
                thought="Apple stock is up 4.06% this week.",
                tool_call=None,
                is_final=True,
            ),
        ],
        "answer": "Apple (AAPL) is currently trading at $189.50 USD, up 4.06% this week (from $182.10).",
    },
    {
        "task_id": "calendar_schedule",
        "task": "Schedule a 1-hour team meeting tomorrow at 2pm and invite alice@example.com.",
        "steps": [
            TraceStep(
                thought="I'll check Alice's availability tomorrow at 2pm first.",
                tool_call=ToolCall("check_availability", {"email": "alice@example.com", "date": "2026-06-16", "time": "14:00", "duration_minutes": 60}, "Available"),
            ),
            TraceStep(
                thought="Alice is available. Creating the meeting.",
                tool_call=ToolCall("create_calendar_event", {"title": "Team Meeting", "date": "2026-06-16", "time": "14:00", "duration_minutes": 60, "attendees": ["alice@example.com"]}, "Event created: ID=evt_8821"),
            ),
            TraceStep(
                thought="Meeting created successfully.",
                tool_call=None,
                is_final=True,
            ),
        ],
        "answer": "Done! I've scheduled a 1-hour team meeting for tomorrow (2026-06-16) at 2:00 PM and invited alice@example.com.",
    },
    {
        "task_id": "code_search",
        "task": "Find all Python files in the project that import 'requests' and show the first match.",
        "steps": [
            TraceStep(
                thought="I'll search the codebase for files importing requests.",
                tool_call=ToolCall("search_files", {"pattern": "import requests", "file_type": "py", "recursive": True}, '["src/api_client.py", "tests/test_api.py", "scripts/fetch_data.py"]'),
            ),
            TraceStep(
                thought="Found 3 files. Let me show the first few lines of the first one.",
                tool_call=ToolCall("read_file", {"path": "src/api_client.py", "lines": 10}, "import requests\nimport json\n\ndef get(url, **kwargs):\n    return requests.get(url, **kwargs)"),
            ),
            TraceStep(
                thought="I have the information needed.",
                tool_call=None,
                is_final=True,
            ),
        ],
        "answer": "Found 3 files importing 'requests': src/api_client.py, tests/test_api.py, scripts/fetch_data.py. The first match (src/api_client.py) uses it in a simple HTTP GET wrapper.",
    },
    {
        "task_id": "database_query",
        "task": "How many users signed up in the last 7 days and what is the average session length?",
        "steps": [
            TraceStep(
                thought="I'll query the database for recent signups.",
                tool_call=ToolCall("run_sql", {"query": "SELECT COUNT(*) as cnt FROM users WHERE created_at >= NOW() - INTERVAL '7 days'"}, '{"cnt": 347}'),
            ),
            TraceStep(
                thought="Now I'll get the average session length for those users.",
                tool_call=ToolCall("run_sql", {"query": "SELECT AVG(session_length_seconds) as avg_session FROM sessions WHERE created_at >= NOW() - INTERVAL '7 days'"}, '{"avg_session": 423.7}'),
            ),
            TraceStep(
                thought="I have both numbers.",
                tool_call=None,
                is_final=True,
            ),
        ],
        "answer": "In the last 7 days: 347 users signed up, with an average session length of 7 minutes 4 seconds (423.7 seconds).",
    },
    {
        "task_id": "translation_pipeline",
        "task": "Translate the product description from English to French and Spanish, then summarize the key features.",
        "steps": [
            TraceStep(
                thought="First I'll translate to French.",
                tool_call=ToolCall("translate", {"text": "High-performance wireless headphones with 40-hour battery life.", "source_lang": "en", "target_lang": "fr"}, "Casque sans fil haute performance avec 40 heures d'autonomie."),
            ),
            TraceStep(
                thought="Now translate to Spanish.",
                tool_call=ToolCall("translate", {"text": "High-performance wireless headphones with 40-hour battery life.", "source_lang": "en", "target_lang": "es"}, "Auriculares inalámbricos de alto rendimiento con 40 horas de batería."),
            ),
            TraceStep(
                thought="All translations done.",
                tool_call=None,
                is_final=True,
            ),
        ],
        "answer": "French: 'Casque sans fil haute performance avec 40 heures d'autonomie.'\nSpanish: 'Auriculares inalámbricos de alto rendimiento con 40 horas de batería.'\nKey features: wireless, high-performance, 40-hour battery.",
    },
    {
        "task_id": "email_summarize",
        "task": "Summarize my unread emails from today and flag any that need urgent attention.",
        "steps": [
            TraceStep(
                thought="I'll fetch today's unread emails.",
                tool_call=ToolCall("list_emails", {"filter": "unread", "date": "today", "max_results": 50}, '[{"id":"e1","from":"boss@co.com","subject":"URGENT: Deploy needed","snippet":"We need the hotfix deployed by 5pm"},{"id":"e2","from":"team@co.com","subject":"Weekly sync notes","snippet":"Notes from this week"},{"id":"e3","from":"noreply@service.com","subject":"Invoice #1042","snippet":"Your invoice is ready"}]'),
            ),
            TraceStep(
                thought="I see 3 emails. The first one is flagged URGENT and needs immediate attention.",
                tool_call=None,
                is_final=True,
            ),
        ],
        "answer": "You have 3 unread emails today.\n\nURGENT (1): 'URGENT: Deploy needed' from boss@co.com — hotfix deploy needed by 5pm.\nNormal (2): Weekly sync notes from team@co.com; Invoice #1042 from noreply@service.com.",
    },
    {
        "task_id": "math_with_tools",
        "task": "Calculate compound interest: $10,000 at 5% annual rate for 10 years, compounded monthly.",
        "steps": [
            TraceStep(
                thought="I'll compute this using the compound interest formula.",
                tool_call=ToolCall("calculate", {"expression": "10000 * (1 + 0.05/12) ** (12*10)"}, "16470.09"),
            ),
            TraceStep(
                thought="The calculation is complete.",
                tool_call=None,
                is_final=True,
            ),
        ],
        "answer": "$10,000 invested at 5% annual interest compounded monthly for 10 years grows to $16,470.09 (a gain of $6,470.09).",
    },
]


def synthetic_tool_tasks(n: Optional[int] = None) -> list[AgentTrace]:
    """Return a list of synthetic reference AgentTraces. No network required.

    Args:
        n: maximum number of tasks to return; None returns all.
    """
    traces = []
    for task_def in _SYNTHETIC_TASKS:
        steps = task_def["steps"]
        trace = AgentTrace(
            task_id=task_def["task_id"],
            task_description=task_def["task"],
            steps=steps,
            final_answer=task_def["answer"],
        )
        traces.append(trace)
    return traces[:n] if n is not None else traces


# ── BFCL source (optional, requires datasets) ─────────────────────────────────

def from_bfcl(
    split: str = "train",
    n: Optional[int] = 50,
    categories: Optional[list[str]] = None,
) -> list[AgentTrace]:
    """Load tasks from the Berkeley Function-Calling Leaderboard dataset.

    Requires: `pip install datasets`
    Dataset: Nexusflow/NexusRaven_API_evaluation (Apache-2.0, 5k+ examples)

    Each example is converted to an AgentTrace with a single tool-call step
    derived from the ground-truth function call annotation.

    Args:
        split: dataset split ("train" is the only available split for BFCL).
        n: max number of traces to load.
        categories: filter by function category (e.g. ["weather", "finance"]).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install 'datasets' to use from_bfcl(): pip install datasets")

    ds = load_dataset("Nexusflow/NexusRaven_API_evaluation", split=split, trust_remote_code=True)

    traces: list[AgentTrace] = []
    for row in ds:
        if n is not None and len(traces) >= n:
            break
        if categories and row.get("category") not in categories:
            continue

        task_id = f"bfcl_{row.get('id', len(traces))}"
        question = row.get("question", row.get("input", ""))
        func_call = row.get("function_call", {})
        tool_name = func_call.get("name", "unknown_tool") if isinstance(func_call, dict) else str(func_call)
        arguments = func_call.get("arguments", {}) if isinstance(func_call, dict) else {}
        expected_output = row.get("expected_output", "")

        tc = ToolCall(tool_name=tool_name, arguments=arguments,
                      result=str(expected_output), step_index=0)
        steps = [
            TraceStep(thought=f"I need to call {tool_name} to answer this.", tool_call=tc),
            TraceStep(thought="Done.", tool_call=None, is_final=True),
        ]
        traces.append(AgentTrace(
            task_id=task_id,
            task_description=question,
            steps=steps,
            final_answer=str(expected_output),
            meta={"source": "bfcl", "category": row.get("category", "")},
        ))

    return traces
