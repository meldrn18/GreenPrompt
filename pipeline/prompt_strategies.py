def zero_shot(text):
    return f"Write a python function to solve the following problem:\n{text}"

def few_shot(text, examples):
    examples_text="\n\n".join(
        [f"Q: {ex['prompt']}\nA: {ex['code']}" for ex in examples]
    )
    return f"{examples_text}\n\nQ: {text}\nA:"

def chain_of_thought(text):
    return f"Write a python function, reasoning step by step to solve this problem:\n{text}"

def reflexion(text):
    return f"First write a python function to solve the given problem, then reflect on it and make improvements:\n{text}"

PROMPT_STRATEGIES = {"zero_shot": zero_shot, "few_shot": few_shot, "chain_of_thought": chain_of_thought, "reflexion": reflexion}


