from __future__ import annotations
import sqlite3
import re
import json
import os
import sys
from typing import Dict, Any, List, Optional

# Ensure project root is in path for models.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server import Environment, create_fastapi_app
from models import MyAction, MyObservation, MyState, ActionType

DB_PATH = "dataclean.db"

# Global variable to store the last created environment instance
_last_env_instance: Optional[DataCleanEnv] = None

# Global shared connection for :memory: persistence across instances
_shared_conn = sqlite3.connect(":memory:", check_same_thread=False)

def _sqlite_regexp_replace(pattern: str, replacement: str, string: str):
    if string is None: return None
    try:
        return re.sub(pattern, replacement, string)
    except Exception:
        return string

_shared_conn.create_function("REGEXP_REPLACE", 3, _sqlite_regexp_replace)
class DataCleanEnv(Environment):
    def __init__(self):
        super().__init__()
        global _last_env_instance
        _last_env_instance = self
        self.conn = _shared_conn
        self.cursor = _shared_conn.cursor() # Get fresh cursor for this instance
        self._env_state = MyState()
    def state(self) -> MyState:
        return self._env_state

    def reset(self, task_id: Optional[str] = "easy_standardization", **kwargs) -> MyObservation:
        self._env_state = MyState(current_task=task_id)
        
        # Wipe DB using shared cursor
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = self.cursor.fetchall()
        for t in tables:
            try:
                self.cursor.execute(f"DROP TABLE IF EXISTS {t[0]}")
            except sqlite3.OperationalError:
                pass
            
        if task_id == "easy_standardization":
            self.cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, date_of_birth TEXT)")
            self.cursor.executemany("INSERT INTO users (id, date_of_birth) VALUES (?, ?)",[
                (1, "01-01-1990"), (2, "05/12/1985"), (3, "1992/08/20"), (4, "12-30-1999")
            ])
            self._env_state.prev_metric = self._grade_easy_task()
        elif task_id == "medium_pii_redaction":
            self.cursor.execute("CREATE TABLE feedback (id INTEGER PRIMARY KEY, customer_message TEXT)")
            self.cursor.executemany("INSERT INTO feedback (id, customer_message) VALUES (?, ?)",[
                (1, "Great service! My card 1234-5678-9012-3456 was charged correctly."),
                (2, "Update billing for 4111222233334444 please.")
            ])
            self._env_state.prev_metric = self._grade_medium_task()
        elif task_id == "hard_entity_resolution":
            self.cursor.execute("CREATE TABLE sales_a (id TEXT, amount REAL, email TEXT)")
            self.cursor.execute("CREATE TABLE sales_b (uuid TEXT, val REAL, user_email TEXT)")
            self.cursor.executemany("INSERT INTO sales_a VALUES (?, ?, ?)", [("A1", 100.0, "john@doe.com")])
            self.cursor.executemany("INSERT INTO sales_b VALUES (?, ?, ?)",[("B1", 100.0, "john@doe.com")])
            self._env_state.prev_metric = self._grade_hard_task()
        elif task_id == "expert_pii_audit":
            self.cursor.execute("CREATE TABLE employees (id INT, full_name TEXT, ssn TEXT, email TEXT)")
            self.cursor.execute("CREATE TABLE server_logs (timestamp TEXT, log_message TEXT)")
            self.cursor.executemany("INSERT INTO employees VALUES (?, ?, ?, ?)", [
                (1, "Alice Smith", "999-00-1111", "alice@company.com"),
                (2, "Bob Jones", "888-11-2222", "bob@ext.com")
            ])
            self.cursor.executemany("INSERT INTO server_logs VALUES (?, ?)", [
                ("2023-01-01 10:00", "User alice@company.com logged in from 192.168.1.1"),
                ("2023-01-01 10:05", "Failed attempt for user unknown@hacker.com"),
                ("2023-01-01 10:10", "Bob Jones (bob@ext.com) updated his profile.")
            ])
            self._env_state.prev_metric = self._grade_expert_task()
        else:
            raise ValueError(f"Unknown task: {task_id}")
            
        self.conn.commit()
        # Sync the initial metric so we only reward IMPROVEMENT
        self._env_state.prev_metric = self._grade_task()
        return self._get_observation("Environment reset successfully.")

    def step(self, action: MyAction) -> MyObservation:
        self._env_state.step_count += 1
        reward = 0.0
        done = False
        metadata = {}
        status = "Success"

        if self._env_state.step_count > self._env_state.max_steps:
            done = True
            metadata["error"] = "max_steps_exceeded"
            status = "Max steps exceeded."
            self._env_state.final_score = self._grade_task()
            obs = self._get_observation(status)
            obs.done = True
            obs.metadata = metadata
            return obs

        try:
            if action.action_type == ActionType.SUBMIT_FINAL:
                self._env_state.final_score = self._grade_task()
                reward = self._env_state.final_score
                done = True
                metadata["final_score"] = self._env_state.final_score
                status = f"Task submitted. Score: {self._env_state.final_score}"
                
            elif action.action_type == ActionType.RUN_SQL_UPDATE:
                query = action.parameters.get("query", "").strip()
                if re.search(r'(?i)\bDROP\s+TABLE\b', query) and self._env_state.current_task != "hard_entity_resolution":
                    reward = -0.5
                    done = True
                    status = "Destructive out-of-scope DROP TABLE detected. Episode terminated."
                    metadata["error"] = "destructive_action"
                else:
                    self.cursor.execute(query)
                    self.conn.commit() # MUTATION COMMIT
                    reward += 0.05

            elif action.action_type == ActionType.APPLY_REGEX_MASK:
                col = action.parameters.get("column")
                pat = action.parameters.get("pattern")
                rep = action.parameters.get("replacement")
                q = f"UPDATE {action.target_table} SET {col} = REGEXP_REPLACE(?, ?, {col})"
                self.cursor.execute(q, (pat, rep))
                self.conn.commit() # MUTATION COMMIT
                reward += 0.05

            elif action.action_type == ActionType.DROP_COLUMN:
                col = action.parameters.get("column")
                self.cursor.execute(f"ALTER TABLE {action.target_table} DROP COLUMN {col}")
                self.conn.commit() # MUTATION COMMIT
                reward += 0.05

            current_metric = self._grade_task()
            self._env_state.final_score = current_metric # Continuous update
            if current_metric > self._env_state.prev_metric:
                reward += (current_metric - self._env_state.prev_metric) * 0.5
                self._env_state.prev_metric = current_metric

        except Exception as e:
            self.conn.rollback()
            status = f"SQL Error: {str(e)}"
            reward -= 0.1

        obs = self._get_observation(status, reward, done, info=metadata)
        print(f"DEBUG SERVER OBS INFO: {obs.info}", flush=True)
        return obs

    def _get_observation(self, status: str, reward: float = 0.0, done: bool = False, info: dict = None) -> MyObservation:
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables =[r[0] for r in self.cursor.fetchall()]
        schema_info = {}
        sample_data = {}
        
        for t in tables:
            try:
                self.cursor.execute(f"PRAGMA table_info({t})")
                schema_info[t] = ", ".join([f"{col[1]} {col[2]}" for col in self.cursor.fetchall()])
                self.cursor.execute(f"SELECT * FROM {t} LIMIT 5")
                cols = [desc[0] for desc in self.cursor.description]
                sample_data[t] =[dict(zip(cols, row)) for row in self.cursor.fetchall()]
            except sqlite3.OperationalError:
                continue
            
        return MyObservation(
            schema_info=schema_info, 
            sample_data=sample_data, 
            last_execution_status=status,
            reward=reward,
            done=done,
            info=info or {}
        )

    def _grade_task(self) -> float:
        if self._env_state.current_task == "easy_standardization":
            return self._grade_easy_task()
        elif self._env_state.current_task == "medium_pii_redaction":
            return self._grade_medium_task()
        elif self._env_state.current_task == "hard_entity_resolution":
            return self._grade_hard_task()
        elif self._env_state.current_task == "expert_pii_audit":
            return self._grade_expert_task()
        return 0.0

    def _grade_expert_task(self) -> float:
        score = 0.0
        try:
            # 1. Check if SSN column was dropped
            self.cursor.execute("PRAGMA table_info(employees)")
            cols = [c[1].lower() for c in self.cursor.fetchall()]
            # print(f"DEBUG EXPERT COLS: {cols}", flush=True)
            if 'ssn' not in cols:
                score += 0.4
            
            # 2. Check if emails in server_logs are masked
            self.cursor.execute("SELECT log_message FROM server_logs")
            logs = self.cursor.fetchall()
            if not logs: return score
            
            # Count logs that no longer have a typical email pattern
            redacted_count = sum(1 for l in logs if not re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", l[0]))
            # print(f"DEBUG REDACTED COUNT: {redacted_count}/{len(logs)}", flush=True)
            score += (redacted_count / len(logs)) * 0.6
                
            return min(1.0, score)
        except Exception as e:
            # print(f"DEBUG EXPERT ERR: {e}", flush=True)
            return 0.0

    def _grade_easy_task(self) -> float:
        try:
            self.cursor.execute("SELECT date_of_birth FROM users")
            rows = self.cursor.fetchall()
            # print(f"DEBUG EASY ROWS: {rows}", flush=True)
            if not rows: return 0.0
            
            # YYYY-MM-DD pattern
            pat = re.compile(r"^\d{4}-\d{2}-\d{2}$")
            correct = sum(1 for r in rows if r[0] and pat.match(str(r[0]).strip()))
            # print(f"DEBUG EASY CORRECT: {correct}/{len(rows)}", flush=True)
            return correct / len(rows)
        except Exception as e:
            # print(f"DEBUG EASY ERR: {e}", flush=True)
            return 0.0

    def _grade_medium_task(self) -> float:
        try:
            self.cursor.execute("SELECT customer_message FROM feedback")
            rows = self.cursor.fetchall()
            expected =[
                "Great service! My card ****-****-****-3456 was charged correctly.",
                "Update billing for ************4444 please."
            ]
            if not rows: return 0.0
            score = 0.0
            for i, r in enumerate(rows):
                if i < len(expected) and r[0] == expected[i]:
                    score += 1.0
            return score / len(expected)
        except Exception:
            return 0.0

    def _grade_hard_task(self) -> float:
        try:
            self.cursor.execute("SELECT email, amount FROM merged_sales")
            rows = set(self.cursor.fetchall())
            golden = {("john@doe.com", 200.0)}
            tp = len(rows.intersection(golden))
            fp = len(rows - golden)
            fn = len(golden - rows)
            if tp == 0: return 0.0
            return (2 * tp) / (2 * tp + fp + fn)
        except Exception:
            return 0.0

# Singleton instance for the environment
_singleton_env: Optional[DataCleanEnv] = None

def get_env() -> DataCleanEnv:
    global _singleton_env, _last_env_instance
    if _singleton_env is None:
        _singleton_env = DataCleanEnv()
    _last_env_instance = _singleton_env # Sync for extra endpoints
    return _singleton_env

# Create FastAPI app using OpenEnv's utility with singleton factory
app = create_fastapi_app(get_env, MyAction, MyObservation)

@app.get("/")
def health_check():
    return {
        "status": "DataCleanEnv is online",
        "documentation": "/docs",
        "hackathon_endpoints": ["/tasks", "/grader", "/baseline"],
        "openenv_endpoints": ["/reset", "/step", "/state"]
    }

# Add extra hackathon endpoints
@app.get("/tasks")
def api_tasks():
    # Return tasks and a detailed JSON schema for the action space
    return {
        "tasks": [
            {"id": "easy_standardization", "name": "Date Standardization", "difficulty": "easy", "objective": "Standardize dates to YYYY-MM-DD."},
            {"id": "medium_pii_redaction", "name": "PII Redaction", "difficulty": "medium", "objective": "Mask credit card numbers in text."},
            {"id": "hard_entity_resolution", "name": "Entity Resolution", "difficulty": "hard", "objective": "Merge sales_a/b into merged_sales table."},
            {"id": "expert_pii_audit", "name": "Expert PII Audit", "difficulty": "expert", "objective": "Mask logs and remove sensitive columns."}
        ],
        "action_schema": {
            "type": "object",
            "properties": {
                "action_type": {"type": "string", "enum": ["RUN_SQL_UPDATE", "APPLY_REGEX_MASK", "DROP_COLUMN", "SUBMIT_FINAL"]},
                "target_table": {"type": "string"},
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "SQL for RUN_SQL_UPDATE"},
                        "column": {"type": "string", "description": "Column name for MASK/DROP"},
                        "pattern": {"type": "string", "description": "Regex for APPLY_REGEX_MASK"},
                        "replacement": {"type": "string", "description": "Mask string for APPLY_REGEX_MASK"}
                    }
                }
            },
            "required": ["action_type", "target_table", "parameters"]
        }
    }


@app.get("/grader")
def api_grader():
    instance = _singleton_env or get_env()
    score = instance._grade_task()
    return {"score": score}

@app.get("/baseline")
def api_baseline():
    import subprocess
    import os
    try:
        # Pass environment variables (like OPENAI_API_KEY / HF_TOKEN) to the subprocess
        env = os.environ.copy()
        result = subprocess.run(["python", "inference.py", "--quiet"], capture_output=True, text=True, env=env)
        return json.loads(result.stdout.splitlines()[-1])
    except Exception as e:
        return {"error": str(e)}

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
