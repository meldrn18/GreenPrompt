# GreenPrompt
This project explores the intersection of prompt engineering and environmental sustainability in large language model (LLM)–based code generation. While prompt design is widely known to influence the accuracy and reasoning of LLM outputs, its impact on the carbon footprint of the generated code remains underexplored.

We hypothesize that different prompting strategies—such as zero-shot, few-shot, chain-of-thought (CoT), and self-reflection (e.g., Reflexion)—not only affect correctness but also influence code complexity, runtime efficiency, and energy consumption. The project aims to systematically evaluate these effects.

Using the MBPP (Mostly Basic Programming Problems) dataset as the benchmark, we will:

Generate solutions for each task using various prompt types.

Measure the execution time, memory usage, and carbon emissions of each generated solution.

Analyze trade-offs between code quality and environmental cost across prompting strategies.

The final deliverable includes:

A labeled dataset of prompt–code–carbon impact triples.

Visual dashboards comparing carbon efficiency across strategies.

Practical recommendations for “greener” prompt engineering in LLM-based programming tools.
