# agent-in-a-week
Demo LangGraph application with MLflow tracing and Databricks deployment

## Overview
This is a LangGraph-based supervisor agent with MLflow tracing and Databricks model serving deployment capabilities.

## Setup

### Prerequisites
- Python 3.11+
- Databricks workspace
- OpenAI API key

### Installation
```bash
cd agents
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file in the `agents` directory:
```bash
OPENAI_API_KEY=your_openai_api_key
DATABRICKS_HOST=https://your-workspace.databricks.com
DATABRICKS_TOKEN=your_databricks_token
```

## Local Development

Run the agent locally with MLflow tracing:
```python
from src.agent_supervisor.deployment import setup_tracing
from src.agent_supervisor.graph import graph

# Enable tracing
setup_tracing()

# Invoke the graph
response = graph.invoke({"messages": [{"role": "user", "content": "Hello"}]})
```

## Deployment

### Manual Deployment
Deploy to Databricks Model Serving:
```bash
cd agents
python deploy.py
```

### CI/CD Deployment
The repository includes a GitHub Actions workflow that automatically deploys to Databricks on push to `main`.

Required GitHub Secrets:
- `DATABRICKS_HOST`: Your Databricks workspace URL
- `DATABRICKS_TOKEN`: Databricks personal access token
- `OPENAI_API_KEY`: OpenAI API key

## Architecture
- **MLflow Tracing**: Automatic tracing of LangGraph execution with `mlflow.langchain.autolog()`
- **Model Registry**: Models logged to MLflow with signatures
- **Databricks Serving**: Deployed as scalable REST endpoints with autoscaling

## References
- [Tracing LangGraph on Databricks](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/integrations/langgraph)
- [LangGraph with Model From Code](https://mlflow.org/blog/langgraph-model-from-code)
