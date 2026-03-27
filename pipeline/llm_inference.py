from openai import OpenAI
import json
from datetime import datetime
import os
from apply_strategy import apply_strategy
from prompt_strategies import PROMPT_STRATEGIES
import re
from multiprocessing import  Queue
from codecarbon import EmissionsTracker
import io
import contextlib
import re, math
from name_enforce import expected_func_name, force_function_name
import subprocess
import sys
import tempfile
from pathlib import Path

RUNNER = r"""
import json, traceback, sys, re, math

payload = json.loads(sys.stdin.read())
code = payload["code"]
tests = payload["tests"]

env = {"re": re, "math": math}

try:
    exec(code, env, env)
    for t in tests:
        exec(t, env, env)
    print("PASS")
    sys.exit(0)
except Exception as e:
    tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    print("FAIL\\n" + tb)
    sys.exit(1)
"""

#configure ai connection
openai_model = "gpt-3.5-turbo" #can be changed to cover more llms
data_path = "data/filtered_dataset.json"
output_path = "generated_code.json"
OpenAI.api_key = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=OpenAI.api_key)

#queue all generated code will be appended to for execution
execution_queue = None
code_fence = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)

def get_execution_queue():
    global execution_queue
    if execution_queue is None:
        execution_queue = Queue()
    return execution_queue

def check_correct(code: str, tests: list, timeout_s: int=3) -> bool:
 with tempfile.TemporaryDirectory() as td:
        runner_path = Path(td) / "runner.py"
        runner_path.write_text(RUNNER, encoding="utf-8")

        payload = {"code": code, "tests": tests}

        try:
            cp = subprocess.run(
                [sys.executable, str(runner_path)],
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            print("failed test: TIMEOUT")
            return False

        if cp.returncode == 0:
            return True
        else:
            print("failed test:\n", cp.stdout)
            return False

#send prompt of mbpp problem to openai, return generated response
def send_prompt(mbpp_problem, prompt_type, model=openai_model, max_tokens=1024):
    func_name = expected_func_name(mbpp_problem["test_list"])
    #format the prompt by strategy 
    prompt = apply_strategy(mbpp_problem,PROMPT_STRATEGIES[prompt_type],func_name=func_name)
    #log inference carbon footprint with EcoLogits
    carbon = EmissionsTracker(output_file="emissions.csv",
        save_to_file=True,
        measure_power_secs=1,
        log_level="error",
        tracking_mode="machine",
    )
    try:
        carbon.start()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role":"user", "content":prompt}],
            max_tokens= max_tokens,
            temperature=0.0,
        )
        emissions = carbon.stop()
        generated_response = response.choices[0].message.content.strip()
        #extracting code from response
        m = code_fence.search(generated_response)
        generated_code = m.group(1).strip() if m else "didnt work"
        if func_name and generated_code != "didnt work":
            generated_code = force_function_name(generated_code, func_name)

        #check response correctness
        if check_correct(generated_code, mbpp_problem["test_list"]):
            #add generated code to queue
            q = get_execution_queue()
            q.put(generated_code)
            print("code correct, added to queue")
        else:
            print("code failed correctness, not added")
            generated_code = None

        return generated_code, emissions
    except Exception as e:
        carbon.stop()
        print(f"error during llm inference: {e}")
        return f"error during llm inference: {e}", None
  



