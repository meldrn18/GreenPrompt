from openai import OpenAI
from ecologits import EcoLogits
import json
from datetime import datetime
import os
from apply_strategy import apply_strategy
from prompt_strategies import PROMPT_STRATEGIES

#configure ai connection
openai_model = "gpt-3.5-turbo" #can be changed to cover more llms
data_path = "data/filtered_dataset.json"
output_path = "generated_code.json"
OpenAI.api_key = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=OpenAI.api_key)

#send prompt of mbpp problem to openai, return generated response
def send_prompt(mbpp_problem, prompt_type, model=openai_model, max_tokens=1024):
    #format the prompt by strategy 
    prompt = apply_strategy(mbpp_problem,PROMPT_STRATEGIES[prompt_type])
    #log inference carbon footprint with EcoLogits
    carbon = EcoLogits()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role":"user", "content":prompt}],
        max_tokens= max_tokens,
        temperature=0.0,
    )
    generated_code = response.choices[0].message.content.strip()
    return generated_code, carbon

