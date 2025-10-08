from openai import OpenAI
from ecologits import EcoLogitsTracker

#send prompt of mbpp problem to openai, return generated response
def send_prompt(mbpp_problem, prompt_type, model="gpt-4o-mini"):
    #log inference carbon footprint with EcoLogits
    with EcoLogitsTracker(api_name="openai"):
        client=OpenAI()
        prompt = prompt_type.format(problem=mbpp_problem)
        response=client.responses.create(model=model,input=prompt)
    return respons.output_text