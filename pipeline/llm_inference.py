from openai import OpenAI
from ecologits import EcoLogits
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

#configure ai connection
openai_model = "gpt-3.5-turbo" #can be changed to cover more llms
data_path = "data/filtered_dataset.json"
output_path = "generated_code.json"
OpenAI.api_key = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=OpenAI.api_key)

#queue all generated code will be appended to for execution
execution_queue = None

def get_execution_queue():
    global execution_queue
    if execution_queue is None:
        execution_queue = Queue()
    return execution_queue

def check_correct(generated_code: str, test_list: list) -> bool:
    local_env = {}
    try:
        exec(generated_code, {}, local_env)
        with contextlib.redirect_stdout(io.StringIO()):
            for test in test_list:
                exec(test,{}, local_env)
        return True
    except Exception as e:
        print(f"failed test: {e}")
        return False

#send prompt of mbpp problem to openai, return generated response
def send_prompt(mbpp_problem, prompt_type, model=openai_model, max_tokens=1024):
    #format the prompt by strategy 
    prompt = apply_strategy(mbpp_problem,PROMPT_STRATEGIES[prompt_type])
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
        match = re.search(r"```(?:python)?\s*(.*?)```", generated_response, re.DOTALL)
        if match:
            generated_code = match.group(1).strip()
        else:
            #if not found return whole response
            #generated_code = generated_response.strip()
            generated_code = "didnt work"
        #check response correctness
        if check_correct(generated_code, mbpp_problem["test_list"]):
            #add generated code to queue
            q = get_execution_queue()
            q.put(generated_code)
            print("code correct, added to queue")
        else:
            print("code failed correctness, not added")

        return generated_code, emissions
    except Exception as e:
        carbon.stop()
        print(f"error during llm inference: {e}")
        return f"error during llm inference: {e}", None
  



