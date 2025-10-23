import json
from llm_inference import send_prompt, execution_queue
from execution_harness import execute_generated_code

def main():
    #load dataset
    with open("data/filtered_dataset.json", "r") as f:
        mbpp = json.load(f)

    results = []
    for i, mbpp_problem in enumerate(mbpp):
        print(f"\n--- Running problem {i+1}/{len(mbpp)} ---")

        # Generate code using LLM
        generated_code, inference_emissions = send_prompt(mbpp_problem, prompt_type="zero_shot")

        if generated_code == "didnt work":
            print(" Code extraction failed.")
            results.append({
                "problem_id": i,
                "status": "failed_extraction",
                "emissions_inference": inference_emissions,
                "emissions_execution": None,
                "output": None
            })
            continue

        # Execute generated code in sandbox
        output, execution_emissions = execute_generated_code(generated_code, timeout=15)
        
        print(" Output:", output)
        print(" Emissions (inference):", inference_emissions)
        print(" Emissions (execution):", execution_emissions)

        results.append({
            "problem_id": i,
            "prompt": mbpp_problem["text"],
            "generated_code": generated_code,
            "output": output,
            "emissions_inference": inference_emissions,
            "emissions_execution": execution_emissions,
        })

    # Save results
    with open("generated_code.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n All problems processed. Results saved to generated_code.json")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support() 
    main()