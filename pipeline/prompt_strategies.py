#add to prompt to return only code
Return_only_code = ("Return ONLY a single python code block fenced like:\n"
                    "```python\n# code here\n```\n"
                    "Do not include any explanation, comments outside the block, or extra text")

#reformat prompt as zero shot
def zero_shot(text, func_name: str | None = None, **_):
    name_hint =f"The function MUST be named exactly '{func_name}'."if func_name else ""
    return f"Write a python function to solve the following problem:\n{text}{name_hint}. {Return_only_code}"

#reformat prompt as few shot
def few_shot(text, examples, func_name: str | None = None, **_):
    examples = examples or []  

    shots = "\n\n".join(
        f"Example:\nProblem: {ex['input']}\nSolution:\n{ex['output']}"
        for ex in examples
    )

    name_hint = f"\nThe function MUST be named exactly `{func_name}`." if func_name else ""

    return (f"{shots}\n\nNow solve this new problem:\n{text}{name_hint}\n\n{Return_only_code}")

#reformat prompt as chain of thought
def chain_of_thought(text, func_name: str | None = None, **_):
    name_hint = f"\nThe function MUST be named exactly `{func_name}`." if func_name else ""

    return f"Write a python function, reasoning step by step to solve this problem:\n{text}{name_hint}. {Return_only_code}"


#reformat prompt as reflexion
def reflexion(text,func_name: str | None = None, **_):
    name_hint = f"\nThe function MUST be named exactly `{func_name}`." if func_name else ""

    return f"First write a python function to solve the given problem, then reflect on it and make improvements:\n{text}{name_hint}. {Return_only_code}"

PROMPT_STRATEGIES = {"zero_shot": zero_shot, "few_shot": few_shot, "chain_of_thought": chain_of_thought, "reflexion": reflexion}


