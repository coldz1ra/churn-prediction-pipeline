.PHONY: setup lint format test train eval predict report

setup:
	python -m pip install -r requirements.txt

lint:
	ruff check .

format:
	black .

test:
	pytest -q

train:
	python -m src.train --config configs/base.yaml

eval:
	python -m src.evaluate --config configs/base.yaml

predict:
	python -m src.inference --input data/scoring.csv --output reports/predictions.csv --config configs/base.yaml

report:
	python -m src.report_gen --config configs/base.yaml
