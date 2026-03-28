import os
import requests
import json
import argparse
import sys
from openai import OpenAI

API_URL = "http://localhost:8000"

def run_task(task_id, model="gpt-4o-mini", max_steps=10):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    try:
        response = requests.post(f"{API_URL}/reset", json={"task_id": task_id})
        res_data = response.json()
        # In OpenEnv /reset returns StepResult structure
        obs = res_data.get("observation", {})
    except Exception as e:
        return 0.0

    done = False
    step = 0
    total_reward = 0.0

    system_prompt = f"""You are a Data Engineering Agent interacting with an OpenEnv SQLite database via JSON actions. 
Your current task ID is: {task_id}.
Respond ONLY with a valid JSON matching the Action schema: 
{{"action_type": "RUN_SQL_UPDATE" | "APPLY_REGEX_MASK" | "DROP_COLUMN" | "SUBMIT_FINAL", "target_table": "table_name", "parameters": {{"query": "SQL", "column": "col", "pattern": "regex", "replacement": "rep"}}}}

Guidelines:
- For easy_standardization: Convert dates to YYYY-MM-DD.
- For medium_pii_redaction: Replace credit card numbers with masks.
- For hard_entity_resolution: Merge sales_a and sales_b into merged_sales (email, amount). Deduplicate by email and sum the amounts.
- For expert_pii_audit: Redact emails in server_logs using regex AND drop the 'ssn' column from employees for compliance.

When you believe the task is complete, use action_type SUBMIT_FINAL."""

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Initial Observation: {json.dumps(obs)}"}]

    final_score = 0.0
    while not done and step < max_steps:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            action_text = response.choices[0].message.content
            action_data = json.loads(action_text)
            
            # OpenEnv requires nesting in 'action' key
            step_res_raw = requests.post(f"{API_URL}/step", json={"action": action_data})
            step_res = step_res_raw.json()
            
            reward = step_res.get("reward", 0.0)
            total_reward += reward
            done = step_res.get("done", False)
            
            # In OpenEnv 'info' is nested in 'observation'
            obs_data = step_res.get("observation", {})
            if done:
                final_score = obs_data.get("info", {}).get("final_score", 0.0)
            
            messages.append({"role": "assistant", "content": action_text})
            messages.append({"role": "user", "content": f"Observation: {json.dumps(obs_data)}\nReward: {reward}"})
            step += 1
        except Exception:
            break

    return final_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    task_ids = ["easy_standardization", "medium_pii_redaction", "hard_entity_resolution", "expert_pii_audit"]
    results = {}
    
    for tid in task_ids:
        score = run_task(tid)
        results[tid] = score

    if args.quiet:
        print(json.dumps(results))
    else:
        print("Baseline Results:")
        for tid, score in results.items():
            print(f"{tid}: {score}")

if __name__ == "__main__":
    main()
