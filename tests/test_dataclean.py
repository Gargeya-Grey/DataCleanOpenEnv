import pytest
from models import MyAction, ActionType, MyObservation

@pytest.fixture
def mock_env():
    from server.app import DataCleanEnv
    return DataCleanEnv()

def test_action_model():
    """Ensure MyAction is correctly structured."""
    action = MyAction(
        action_type=ActionType.RUN_SQL_UPDATE,
        target_table="users",
        parameters={"query": "UPDATE users SET date_of_birth = '1990-01-01'"}
    )
    assert action.action_type == ActionType.RUN_SQL_UPDATE

def test_observation_model():
    """Ensure MyObservation is correctly structured."""
    obs = MyObservation(
        schema_info={"users": "id INT, name TEXT"},
        sample_data={"users":[{"id": 1, "name": "Test"}]},
        last_execution_status="Success"
    )
    assert "users" in obs.schema_info

def test_easy_task_grader(mock_env):
    """Validate deterministic 0.0 - 1.0 grader for standardizing dates."""
    mock_env.reset("easy_standardization")
    
    # Simulate perfect formatting
    mock_env.cursor.execute("UPDATE users SET date_of_birth = '1990-01-01'")
    score = mock_env._grade_easy_task()
    assert score == 1.0

    # Simulate partial correctness
    mock_env.cursor.execute("UPDATE users SET date_of_birth = '01/01/1990' WHERE id = 1")
    score = mock_env._grade_easy_task()
    assert 0.0 < score < 1.0

def test_destructive_action_penalty(mock_env):
    """Ensure out-of-scope destructive actions terminate episode with penalty."""
    mock_env.reset("easy_standardization")
    action = MyAction(
        action_type=ActionType.RUN_SQL_UPDATE,
        target_table="users",
        parameters={"query": "DROP TABLE users;"}
    )
    response = mock_env.step(action)
    assert response.reward == -0.5
    assert response.done is True
    assert "destructive" in response.info["error"].lower()

def test_medium_task_grader(mock_env):
    """Validate PII redaction grader."""
    mock_env.reset("medium_pii_redaction")
    
    # Simulate perfect redaction
    mock_env.cursor.execute("UPDATE feedback SET customer_message = 'Great service! My card ****-****-****-3456 was charged correctly.' WHERE id = 1")
    mock_env.cursor.execute("UPDATE feedback SET customer_message = 'Update billing for ************4444 please.' WHERE id = 2")
    score = mock_env._grade_medium_task()
    assert score == 1.0

def test_hard_task_grader(mock_env):
    """Validate entity resolution grader."""
    mock_env.reset("hard_entity_resolution")
    
    # Simulate correct merge
    mock_env.cursor.execute("CREATE TABLE merged_sales (email TEXT, amount REAL)")
    mock_env.cursor.execute("INSERT INTO merged_sales VALUES ('john@doe.com', 200.0)")
    score = mock_env._grade_hard_task()
    assert score == 1.0

def test_expert_task_grader(mock_env):
    """Validate cross-table PII audit grader."""
    mock_env.reset("expert_pii_audit")
    
    # 1. Drop SSN
    mock_env.cursor.execute("ALTER TABLE employees DROP COLUMN ssn")
    # 2. Redact emails in logs
    mock_env.cursor.execute("UPDATE server_logs SET log_message = 'REDACTED'")
    
    score = mock_env._grade_expert_task()
    assert score == 1.0
