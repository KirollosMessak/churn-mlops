# Shortcuts. On Windows without make, run the commands after each target directly.
export PYTHONPATH := src

install:      ## install dependencies
	pip install -r requirements.txt

train:        ## validate data, train candidates, quality gate, promote
	python -m churn.train

serve:        ## run the API at http://localhost:8000/docs
	uvicorn churn.api:app --reload

test:         ## lint + unit tests
	ruff check . && pytest -q

simulate:     ## send drifted production traffic through the API
	python -m churn.monitoring.simulate --scenario drift --n 500

drift:        ## drift report -> reports/drift_report.html
	python -m churn.monitoring.drift --window 500

mlflow-ui:    ## browse experiments at http://localhost:5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db

docker-up:    ## API + MLflow server in containers
	docker compose up --build

.PHONY: install train serve test simulate drift mlflow-ui docker-up
