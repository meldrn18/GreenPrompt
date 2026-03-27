import re

FUNC_RE = re.compile(r"assert\s+([A-Za-z_]\w*)\s*\(")

def expected_func_name(test_list):
    for t in test_list:
        m = FUNC_RE.search(t)
        if m:
            return m.group(1)
    return None

DEF_RE = re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", flags=re.MULTILINE)

def force_function_name(code: str, expected: str) -> str:
    if not expected:
        return code
    # replace the first def name with expected
    return DEF_RE.sub(f"def {expected}(", code, count=1)
