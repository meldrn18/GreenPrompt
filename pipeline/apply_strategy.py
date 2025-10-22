from prompt_strategies import PROMPT_STRATEGIES
from typing import Callable, Dict, Optional

def apply_strategy(mbpp,strategy:Callable, examples: Optional[list]=None
        )-> str:
    if strategy== PROMPT_STRATEGIES["few_shot"]:
        return strategy(mbpp["text"], examples=examples)
    else:
        return  strategy(mbpp["text"])

