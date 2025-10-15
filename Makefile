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

data_telco:
	python scripts/get_telco.py data/train.csv

train_telco: data_telco
	python -m src.train --config configs/telco.yaml

eval_telco:
	python -m src.evaluate --config configs/telco.yaml

report_telco:
	python -m src.report_gen --config configs/telco.yaml

	python -m src.report_gen --config configs/base.yaml

# --- Docker & Demo ---
docker_build:
	docker build -t churn-api:latest .
docker_run:
	docker compose up --build
api_demo:
	bash scripts/api_demo.sh

api_start:
	uvicorn src.app:app --host 0.0.0.0 --port 8000
api_demo:
	bash scripts/api_demo.sh
train_all:
	make data_telco && python -m src.train --config configs/telco.yaml && python -m src.evaluate --config configs/telco.yaml && python -m src.report_gen --config configs/telco.yaml && python scripts/update_docs.py
