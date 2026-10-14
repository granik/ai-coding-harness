.PHONY: setup prepare-demo docker-image test demo-scripted run

setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install -e '.[demo]'

prepare-demo:
	./scripts/prepare_demo.sh

docker-image:
	docker build -t poc-harness-python:3.11 -f docker/Dockerfile .

test:
	.venv/bin/python -m unittest discover -s tests -v

demo-scripted:
	.venv/bin/python -m coding_harness run --config harness.local.toml --scripted demo/scripted_repair.jsonl

run:
	.venv/bin/python -m coding_harness run --config harness.example.toml

