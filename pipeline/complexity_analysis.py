from radon.complexity import cc_visit
from radon.metrics import mi_visit, h_visit
from radon.raw import analyze

def complexity_analysis(code):
    #calc cyclomatic complexity
    cc_results = cc_visit(code)
    cc_values = [block.complexity for block in cc_results]
    max_cc = max(cc_values) if cc_values else 0
    mean_cc = sum(cc_values)/len(cc_values) if cc_values else 0
    #calc maintainability index
    mi = mi_visit(code, multi=False)
    # Halstead metrics 
    h = h_visit(code)
    # if total exists, use it; otherwise use h directly
    h_total = getattr(h, "total", h)  
    volume = None
    for attr in ("volume", "vol", "V"):
        if hasattr(h_total, attr):
            volume = getattr(h_total, attr)
            break
    #LOC metrics
    raw = analyze(code)
    return {
        "functions_analyzed": len(cc_results),
        "max_cyclomatic_complexity": max_cc,
        "mean_cyclomatic_complexity": mean_cc,
        "maintainability_index": mi,
        "halstead_volume": round(volume, 2) if volume is not None else None,
        "lloc": raw.lloc,
        "sloc": raw.sloc,
        "comments": raw.comments,
    } 