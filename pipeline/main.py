import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.getLogger("codecarbon").setLevel(logging.ERROR)
import json
import os
import multiprocessing
import time
from datetime import datetime
from llm_inference import send_prompt, execution_queue
from execution_harness import execute_generated_code
from prompt_strategies import PROMPT_STRATEGIES
from complexity_analysis import complexity_analysis


dataset_path = "data/merged_dataset.json"
results_path = "data/final_results.jsonl"
strategies = list(PROMPT_STRATEGIES.keys())
limit = None
sleep_between = 0.5


def main():
    #load dataset
    with open(dataset_path, "r") as f:
        mbpp = json.load(f)
    if limit:
        mbpp = mbpp[:limit]

    for i, mbpp_problem in enumerate(mbpp):
        print(f"\n--- Problem {i+1}/{len(mbpp)} ---")
        prompt_text = mbpp_problem["text"]
        for strategy_name, strategy_fn in PROMPT_STRATEGIES.items():
            print(f"-----using strategy: {strategy_name} ------")
            try:
                #generate code using LLM
                generated_code, inference_emissions = send_prompt(mbpp_problem, prompt_type=strategy_name)
                if not generated_code or generated_code == "didnt work":
                    print(" Skipped. code not correct or unavailable.")
                    continue
            except Exception as e:
                print(f"error during inference: {e}")
                continue
            #complexity analysis
            try:
                complexity_stats = complexity_analysis(generated_code)
            except Exception as e:
                #if code invalid
                complexity_stats = {"analysis_error": str(e)}
                print(f"Complexity analysis failed:{e}")
                continue

            try:
                #execute generated code in sandbox
                output, execution_emissions, memory_mb = execute_generated_code(generated_code, timeout=15)
            
                print(" Output:", output)
                print(" Emissions   CO2:", execution_emissions)
                print(" Memory:", memory_mb, "MB")
            except Exception as e:
                print(f"error during execution: {e}")
                execution_emissions, memory_mb = None, None
                output = str(e)
            total_carbon = {"inference": inference_emissions, "execution":execution_emissions}

            print(f"done.")
            record = {
                "problem_id": 187+ i,
                "strategy": strategy_name,
                "prompt": prompt_text,
                "generated_code": generated_code,
                "output": output,
                "carbon": total_carbon,
                "memory_mb": memory_mb,
                "complexity":complexity_stats,
            }
            #save results
            with open(results_path, "a") as f:
                f.write(json.dumps(record)+"\n")
            print("saved result")

    print("\n All problems processed. Results saved to {results_path}")

if __name__ == "__main__":
    multiprocessing.freeze_support() 
    main()