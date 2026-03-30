
# Production LLM Project

## Run locally
pip install -r requirements.txt
uvicorn api.main:app --reload

## Docker
docker-compose up --build
