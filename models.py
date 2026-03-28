from enum import Enum
from typing import Dict, Any, List, Optional
from openenv.core.env_server import Action, Observation, State
from pydantic import Field

class ActionType(str, Enum):
    RUN_SQL_UPDATE = "RUN_SQL_UPDATE"
    APPLY_REGEX_MASK = "APPLY_REGEX_MASK"
    DROP_COLUMN = "DROP_COLUMN"
    SUBMIT_FINAL = "SUBMIT_FINAL"

class MyAction(Action):
    action_type: ActionType
    target_table: str
    parameters: Dict[str, Any]

class MyObservation(Observation):
    schema_info: Dict[str, str]
    sample_data: Dict[str, List[Dict[str, Any]]]
    last_execution_status: str
    reward: float = 0.0
    done: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self):
        return {
            "schema_info": self.schema_info,
            "sample_data": self.sample_data,
            "last_execution_status": self.last_execution_status,
            "reward": self.reward,
            "done": self.done,
            "metadata": self.metadata
        }

class MyState(State):
    current_task: Optional[str] = None
    step_count: int = 0
    prev_metric: float = 0.0
    max_steps: int = 15
    # Remove static final_score to avoid stale data
