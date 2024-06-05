import mlflow
from typing import Any

def create_mlflow_experiment(experiment_name:str) -> str:
    """
    Create a new mlflow experiment with the given name and artifact location
    """
    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name
        )

    except:
        print(f"Experiment {experiment_name} already exists")
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    return experiment_id


def get_mlflow_experiment(experiment_id:str=None, experiment_name:str=None) -> mlflow.entities.Experiment:
    """
    Retrieves the mlflow experiment with the given name or id
    """

    if experiment_id is not None:
        experiment = mlflow.get_experiment(experiment_id)
    elif experiment_name is not None:
        experiment = mlflow.get_experiment_by_name(experiment_name)
    else:
        raise ValueError("Either experiment id or experiment name must be given")
    return experiment