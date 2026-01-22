#!/usr/bin/env python3
"""Deploy LangGraph agent to Databricks Model Serving."""

import os
import sys
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput

from src.agent_supervisor.deployment import log_model


def deploy_to_databricks(
    model_name: str = "agent-supervisor",
    endpoint_name: str = "agent-supervisor-endpoint",
    workload_size: str = "Small",
):
    """Deploy the model to Databricks Model Serving.

    Args:
        model_name: Name of the registered model in MLflow
        endpoint_name: Name for the serving endpoint
        workload_size: Size of the workload (Small, Medium, Large)
    """
    # Initialize Databricks client (uses DATABRICKS_HOST and DATABRICKS_TOKEN env vars)
    w = WorkspaceClient()

    # Get the latest version of the model
    client = mlflow.tracking.MlflowClient()
    model_version = client.get_latest_versions(model_name, stages=["None"])[0].version

    print(f"Deploying model {model_name} version {model_version} to endpoint {endpoint_name}")

    # Create or update serving endpoint
    try:
        w.serving_endpoints.create(
            name=endpoint_name,
            config=EndpointCoreConfigInput(
                served_entities=[
                    ServedEntityInput(
                        entity_name=model_name,
                        entity_version=model_version,
                        workload_size=workload_size,
                        scale_to_zero_enabled=True,
                    )
                ]
            ),
        )
        print(f"Created endpoint: {endpoint_name}")
    except Exception as e:
        if "already exists" in str(e):
            # Update existing endpoint
            w.serving_endpoints.update_config(
                name=endpoint_name,
                served_entities=[
                    ServedEntityInput(
                        entity_name=model_name,
                        entity_version=model_version,
                        workload_size=workload_size,
                        scale_to_zero_enabled=True,
                    )
                ],
            )
            print(f"Updated endpoint: {endpoint_name}")
        else:
            raise


if __name__ == "__main__":
    import mlflow

    # Log the model to MLflow
    print("Logging model to MLflow...")
    model_info = log_model()
    print(f"Model logged: {model_info.model_uri}")

    # Deploy to Databricks
    print("\nDeploying to Databricks Model Serving...")
    deploy_to_databricks()
    print("Deployment complete!")
