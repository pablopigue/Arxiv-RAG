.PHONY: install index ask api ui eval evalset test

install:
	pip install -r requirements.txt

# Build the index. Override QUERY/MAX on the command line.
QUERY ?= cat:cs.LG AND abs:retrieval augmented generation
MAX ?= 30
index:
	python scripts/build_index.py --query "$(QUERY)" --max $(MAX)

ask:
	python scripts/ask.py "$(Q)"

api:
	uvicorn src.api.main:app --reload

ui:
	streamlit run src/app/streamlit_app.py

evalset:
	python eval/eval_dataset.py

eval:
	python eval/evaluate.py --faithfulness

test:
	pytest -q
