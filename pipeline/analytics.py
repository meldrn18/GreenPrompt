import pandas as pd

def test_correctness(output, expected):
    return output.strip()==expected.strip()

def overall_results(results):
    df=pd.DataFrame(results)
    return df.describe()