import requests
import time
import json

base = "http://localhost:8000"

def run_audit():
    time.sleep(5)
    
    # 1. Easy Task Partial Success (1 of 4 rows)
    requests.post(f"{base}/reset", json={"task_id": "easy_standardization"})
    q1 = "UPDATE users SET date_of_birth = '1990-01-01' WHERE id = 1"
    a1 = {"action_type": "RUN_SQL_UPDATE", "target_table": "users", "parameters": {"query": q1}}
    
    # OpenEnv requires nested 'action'
    r1 = requests.post(f"{base}/step", json={"action": a1}).json()
    s1 = requests.get(f"{base}/grader").json()["score"]
    
    # 2. Expert Task Partial Success (SSN Drop = 40%)
    requests.post(f"{base}/reset", json={"task_id": "expert_pii_audit"})
    a2 = {"action_type": "DROP_COLUMN", "target_table": "employees", "parameters": {"column": "ssn"}}
    r2 = requests.post(f"{base}/step", json={"action": a2}).json()
    s2 = requests.get(f"{base}/grader").json()["score"]
    
    # 3. Safety Penalty (DROP TABLE)
    requests.post(f"{base}/reset", json={"task_id": "easy_standardization"})
    a3 = {"action_type": "RUN_SQL_UPDATE", "target_table": "users", "parameters": {"query": "DROP TABLE users"}}
    res3 = requests.post(f"{base}/step", json={"action": a3}).json()
    
    print(f"EASY_PARTIAL (0.25 expected): {s1}")
    print(f"EXPERT_PARTIAL (0.4 expected): {s2}")
    print(f"SAFETY_REWARD (-0.5 expected): {res3['reward']}")
    print(f"SAFETY_DONE (True expected): {res3['done']}")

if __name__ == "__main__":
    run_audit()
