# Data Agent (POC)

This repository provides the engine.
The real value emerges when you connect it to your own long-term thinking logs.

A minimal data agent that answers natural language questions over CSV/JSON/SQLite data. It follows the design direction from OpenAI's in-house data agent article by emphasizing schema-first context, layered enrichment, and a path to retrieval-powered self-improvement.

## Design intent
- **Natural language -> plan -> query -> summary**: questions are analyzed into a structured plan, executed against data, and summarized.
- **Schema-first context (Codex Enrichment, simplified)**: table/column metadata and samples are fed into the planner so it can infer which tables/fields to use.
- **Self-improvement entry point**: every interaction is logged to JSONL, and retrieval interfaces are stubbed for future embeddings + memory.
- **Modular and extensible**: data loading, planning, querying, logging, and retrieval interfaces are separated into modules.

## Quick start
1) Install dependencies:

```
pip install -r data_agent/requirements.txt
```

2) Set your OpenAI API key (and optional model):

```
export OPENAI_API_KEY="your_key"
export OPENAI_MODEL="gpt-4.1-mini"
```

3) Run the CLI:

```
python -m data_agent.main --data /path/to/data.csv --question "Top 5 customers by revenue"
```

Interactive mode (no `--question`) will prompt for questions.

## File structure
```
data_agent/
├─ main.py
├─ agent.py
├─ data_loader.py
├─ retrieval.py
├─ log_store.py
├─ prompts/
│ ├─ analyze_base.txt
│ ├─ context_schema_template.txt
│ └─ future_reflection_template.txt
├─ requirements.txt
├─ README.md
```

## Notes on behavior
- CSV/JSON files are loaded into memory via pandas. SQLite tables are queried directly.
- The planner expects one table per query (no joins yet). This keeps execution simple and safe.
- Logs are stored as JSONL with: question, used tables, plan, summary, preview, timestamp.

## Future direction
- **Embeddings + retrieval**: `retrieval.py` provides a backend protocol for similarity search so logs and schema context can be re-ranked and reused.
- **Context layering**: enrich the planner context with table usage stats, prior successful plans, and dataset-specific notes.
- **Self-improvement loop**: use the log store + retrieval context to adjust future plans and reduce errors over time.

## Limitations
- No joins or multi-step workflows yet.
- Planner is minimal; fallback heuristics are used if no API key is present.
- Large CSV/JSON files may require optimization (sampling or chunked queries).
