.PHONY: install index ask agent api ui eval evalset overnight test lint format

install:
	uv sync

# Build the index. Override QUERY/MAX on the command line.
QUERY ?= cat:cs.LG AND abs:retrieval augmented generation
MAX ?= 30
index:
	uv run scripts/build_index.py --query "$(QUERY)" --max $(MAX)

ask:
	uv run scripts/ask.py "$(Q)"

# Corrective-RAG agent: may fetch and index new arXiv papers.
agent:
	uv run scripts/agent.py "$(Q)"

api:
	uv run uvicorn src.api.main:app --reload

ui:
	uv run streamlit run src/app/streamlit_app.py

evalset:
	uv run eval/eval_dataset.py

eval:
	uv run eval/evaluate.py --faithfulness

# Long run (a few hours): 200-question set, all 4 retrieval configs, CIs, saved to JSON.
overnight:
	uv run eval/eval_dataset.py --n 200 --out eval/qa_set_200.jsonl
	PYTHONUNBUFFERED=1 uv run eval/evaluate.py --qa eval/qa_set_200.jsonl --hybrid --faithfulness \
		--save eval/results_200.json 2>&1 | tee eval/results_200.txt

test:
	uv run pytest -q

# Ruff: `make lint` checks, `make format` rewrites files in place.
lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .
