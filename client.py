from openenv.core.env_client import EnvClient, StepResult
from models import MyAction, MyObservation, MyState
from typing import Optional, Dict, Any

class DataCleanEnvClient(EnvClient[MyAction, MyObservation, MyState]):
    """
    Client-side implementation for DataCleanEnv.
    Handles serialization and parsing of environment interactions.
    """

    def _step_payload(self, action: MyAction) -> dict:
        """Serialize MyAction into a dictionary for the server."""
        return {
            "action_type": action.action_type,
            "target_table": action.target_table,
            "parameters": action.parameters
        }

    def _parse_result(self, payload: dict) -> StepResult[MyObservation]:
        """Parse server's JSON response into a StepResult object."""
        # The payload from create_fastapi_app standard /step endpoint
        # typically contains 'observation', 'reward', 'done', and 'info'
        obs_data = payload.get("observation", {})
        observation = MyObservation(
            schema_info=obs_data.get("schema_info", {}),
            sample_data=obs_data.get("sample_data", {}),
            last_execution_status=obs_data.get("last_execution_status", ""),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
            metadata=payload.get("metadata", {})
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False)
        )

    def _parse_state(self, payload: dict) -> MyState:
        """Parse server's JSON response into a MyState object."""
        return MyState(
            current_task=payload.get("current_task"),
            step_count=payload.get("step_count", 0),
            prev_metric=payload.get("prev_metric", 0.0),
            max_steps=payload.get("max_steps", 15),
            final_score=payload.get("final_score", 0.0)
        )
