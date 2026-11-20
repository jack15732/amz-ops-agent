PYTHON ?= python3
EMBEDDING_PROVIDER ?= huggingface
EMBEDDING_MODEL ?= moka-ai/m3e-base
export PYTHONPATH := src

.PHONY: install test eval eval-hf eval-hash serve ui ingest clean

install:
	$(PYTHON) -m pip install -e ".[ui,hf,eval,dev]"

test:
	$(PYTHON) -m pytest -q

eval:
	AMZ_QDRANT_MODE=memory AMZ_AUTO_INGEST=true \
	$(PYTHON) -m amz_agent.cli eval --suite all \
		--embedding-provider $(EMBEDDING_PROVIDER) --embedding-model $(EMBEDDING_MODEL)

eval-hf:
	$(MAKE) eval EMBEDDING_PROVIDER=huggingface

eval-hash:
	$(MAKE) eval EMBEDDING_PROVIDER=hash

serve:
	$(PYTHON) -m amz_agent.cli serve --port 8000

ui:
	streamlit run apps/streamlit_app.py

ingest:
	$(PYTHON) -m amz_agent.cli ingest data/knowledge --reset

clean:
	rm -rf data/runtime
