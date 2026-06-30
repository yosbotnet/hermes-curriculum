# Thin wrappers over the `curriculum` CLI for the common build/serve loop.
# See AGENTS.md for the full bootstrap, env vars, and troubleshooting.
# Override on the command line: `make build CORPUS=other.json`,
# `make status CURRICULUM=/path/to/curriculum`.

CURRICULUM ?= .venv/bin/curriculum
CORPUS ?= corpus.json

.PHONY: db install build link questions status mcp reset test

# Start Postgres+pgvector on :5433 (schema/001_init.sql auto-applied on first boot).
db:
	docker compose up -d db

# Create the venv and install the package with the postgres+mcp+nous adapters.
install:
	uv venv && uv pip install -e '.[postgres,mcp,nous]'

# Full pipeline for $(CORPUS): ingest -> link -> questions (course from the manifest).
build:
	$(CURRICULUM) build $(CORPUS)

# Link isolated concepts via embedding-guided edge repair (default course).
link:
	$(CURRICULUM) link

# Generate exam questions over the persisted graph (default course).
questions:
	$(CURRICULUM) questions

# Read-only graph counts for the default course (no inference, no writes).
status:
	$(CURRICULUM) status

# Register the MCP server with Hermes (or print the command if hermes is absent).
mcp:
	$(CURRICULUM) mcp-register

# Wipe all graph/state tables and delete the OKF bundle (./bundle by default).
reset:
	docker compose exec -T db psql -U curriculum -d curriculum -c 'TRUNCATE concept, edge, question, learner_state, review_log, course_profile CASCADE;'
	rm -rf bundle

# Run the stdlib unittest suite (needs no database and no API key).
test:
	PYTHONPATH=src python3 -m unittest discover -s tests -t .
