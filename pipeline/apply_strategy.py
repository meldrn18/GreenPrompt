from prompt_strategies import PROMPT_STRATEGIES
from typing import Callable, Dict, Optional

def apply_strategy(text,strategy:Callable, examples: Optional[list]=None
        )-> str:
    if strategy== PROMPT_STRATEGIES["few_shot"]:
        return strategy(text, examples=examples)
    else:
        return  strategy(text)

