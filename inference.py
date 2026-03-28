import os
import json
import asyncio
import argparse
from typing import List, Dict, Any
from openai import OpenAI
from client import DataCleanEnvClient
from models import MyAction, MyObservation, MyState, ActionType

# Required Environment Variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

# Environment Config
ENV_URL = "http://localhost:8000"

async def run_task(task_id: str, client: OpenAI, env_client: DataCleanEnvClient, max_steps: int = 10) -> float:
    print(f"--- Starting Task: {task_id} ---")
    
    # 1. Reset Environment
    try:
        reset_res = await env_client.reset(task_id=task_id)
        obs = reset_res.observation
    except Exception as e:
        print(f"Reset failed: {e}")
        return 0.0

    done = False
    step = 0
    final_score = 0.0

    system_prompt = f"""You are a Data Engineering Agent. Current task: {task_id}.
Respond ONLY with a valid JSON matching the Action schema:
{{
    "action_type": "RUN_SQL_UPDATE" | "APPLY_REGEX_MASK" | "DROP_COLUMN" | "SUBMIT_FINAL",
    "target_table": "table_name",
    "parameters": {{
        "query": "SQL",
        "column": "col_name",
        "pattern": "regex",
        "replacement": "string"
    }}
}}

Guidelines:
- easy_standardization: Convert dates to YYYY-MM-DD.
- medium_pii_redaction: Mask credit card numbers.
- hard_entity_resolution: Merge sales_a and sales_b into merged_sales (email, amount).
- expert_pii_audit: Redact emails in server_logs AND drop 'ssn' from employees.

When finished, respond with action_type SUBMIT_FINAL."""

    # Convert Observation to dict for JSON serialization
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Initial Observation: {json.dumps(obs.to_dict())}"}
    ]

    while not done and step < max_steps:
        try:
            # 2. Get Model Action
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            action_text = completion.choices[0].message.content
            action_data = json.loads(action_text)
            
            # 3. Environment Step
            action = MyAction(**action_data)
            step_result = await env_client.step(action)
            
            obs = step_result.observation
            reward = step_result.reward
            done = step_result.done
            
            print(f"Step {step+1}: Reward {reward}, Done: {done}")
            
            if done:
                final_score = obs.info.get("final_score", 0.0)
            
            messages.append({"role": "assistant", "content": action_text})
            messages.append({"role": "user", "content": f"Observation: {json.dumps(obs.to_dict())}\nReward: {reward}"})
            step += 1
            
        except Exception as e:
            print(f"Step {step+1} Error: {e}")
            break

    print(f"Task {task_id} Completed. Final Score: {final_score}")
    return final_score

async def async_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not API_KEY:
        print("Error: HF_TOKEN or OPENAI_API_KEY environment variable not set.")
        return

    # Initialize Clients
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env_client = DataCleanEnvClient(base_url=ENV_URL)

    task_ids = ["easy_standardization", "medium_pii_redaction", "hard_entity_resolution", "expert_pii_audit"]
    results = {}
    
    for tid in task_ids:
        score = await run_task(tid, client, env_client)
        results[tid] = score

    # Clean up client
    await env_client.close()

    if args.quiet:
        print(json.dumps(results))
    else:
        print("\n" + "="*30)
        print("FINAL BASELINE RESULTS")
        print("="*30)
        for tid, score in results.items():
            print(f"{tid:25}: {score}")

if __name__ == "__main__":
    asyncio.run(async_main())
