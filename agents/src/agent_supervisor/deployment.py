"""Deployment utilities for LangGraph agent with MLflow tracing and Databricks model serving."""

import mlflow
from mlflow.models import infer_signature

from src.agent_supervisor.graph import graph


def setup_tracing():
    """Enable MLflow automatic tracing for LangChain/LangGraph."""
    mlflow.langchain.autolog()


def log_model(
    model_name: str = "agent-supervisor",
    experiment_name: str = "/agent-in-a-week",
):
    """Log the LangGraph model to MLflow with tracing enabled.

    Args:
        model_name: Name for the registered model
        experiment_name: MLflow experiment name

    Returns:
        Model info from MLflow
    """
    # Enable tracing
    setup_tracing()

    # Set experiment
    mlflow.set_experiment(experiment_name)

    # Start MLflow run
    with mlflow.start_run() as run:
        # Infer signature from a sample input/output
        sample_input = {"messages": [{"role": "user", "content": "Hello"}]}
        sample_output = graph.invoke(sample_input)
        signature = infer_signature(sample_input, sample_output)

        # Log the model using model-from-code
        model_info = mlflow.langchain.log_model(
            lc_model=graph,
            artifact_path="model",
            model_name=model_name,
            signature=signature,
        )

        return model_info
