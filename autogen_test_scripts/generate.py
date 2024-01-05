#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import pdb
import torch
import intel_extension_for_pytorch as ipex
import time
import argparse

from bigdl.llm.transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer
from transformers.generation.configuration_utils import GenerationConfig

# you could tune the prompt based on your own model,
# here the prompt tuning refers to https://huggingface.co/georgesung/llama2_7b_chat_uncensored#prompt-style
DEFAULT_SYSTEM_PROMPT = """\
"""

def get_prompt(message: str, chat_history: list[tuple[str, str]],
               system_prompt: str) -> str:
    texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    # The first user input is _not_ stripped
    do_strip = False
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
    message = message.strip() if do_strip else message
    texts.append(f'{message} [/INST]')
    return ''.join(texts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Llama2 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-chat-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--n-predict', type=int, default=300,
                        help='Max tokens to predict')
    parser.add_argument('--device', type=str, default='cpu', help='device')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    kwargs = {'torch_dtype': torch.bfloat16, 'revision': 'main'}
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                load_in_4bit=False,
                                                low_cpu_mem_usage=True,
                                                use_cache=True,
                                                **kwargs)
    
    ###############   XPU Device  #####################
    if args.device == 'xpu':
        model = model.to('xpu')
    ###############   CPU Device  #################

    # Load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # define the autogen prompt
    # solve equality
    prompt = "You are a helpful assistant.\n### Human: Got any creative ideas for a 10 year old’s birthday?\n### Assistant: Of course! Here are some creative ideas for a 10-year-old's birthday party:\n1. Treasure Hunt: Organize a treasure hunt in your backyard or nearby park. Create clues and riddles for the kids to solve, leading them to hidden treasures and surprises.\n2. Science Party: Plan a science-themed party where kids can engage in fun and interactive experiments. You can set up different stations with activities like making slime, erupting volcanoes, or creating simple chemical reactions.\n3. Outdoor Movie Night: Set up a backyard movie night with a projector and a large screen or white sheet. Create a cozy seating area with blankets and pillows, and serve popcorn and snacks while the kids enjoy a favorite movie under the stars.\n4. DIY Crafts Party: Arrange a craft party where kids can unleash their creativity. Provide a variety of craft supplies like beads, paints, and fabrics, and let them create their own unique masterpieces to take home as party favors.\n5. Sports Olympics: Host a mini Olympics event with various sports and games. Set up different stations for activities like sack races, relay races, basketball shooting, and obstacle courses. Give out medals or certificates to the participants.\n6. Cooking Party: Have a cooking-themed party where the kids can prepare their own mini pizzas, cupcakes, or cookies. Provide toppings, frosting, and decorating supplies, and let them get hands-on in the kitchen.\n7. Superhero Training Camp: Create a superhero-themed party where the kids can engage in fun training activities. Set up an obstacle course, have them design their own superhero capes or masks, and organize superhero-themed games and challenges.\n8. Outdoor Adventure: Plan an outdoor adventure party at a local park or nature reserve. Arrange activities like hiking, nature scavenger hunts, or a picnic with games. Encourage exploration and appreciation for the outdoors.\nRemember to tailor the activities to the birthday child's interests and preferences. Have a great celebration!\n### Human: Let's use Python to solve a math problem.\n\nQuery requirements:\nYou should always use the 'print' function for the output and use fractions/radical forms instead of decimals.\nYou can use packages like sympy to help you.\nYou must follow the formats below to write your code:\n```python\n# your code\n```\n\nFirst state the key idea to solve the problem. You may choose from three ways to solve the problem:\nCase 1: If the problem can be solved with Python code directly, please write a program to solve it. You can enumerate all possible arrangements if needed.\nCase 2: If the problem is mostly reasoning, you can solve it by yourself directly.\nCase 3: If the problem cannot be handled in the above two ways, please follow this process:\n1. Solve the problem step by step (do not over-divide the steps).\n2. Take out any queries that can be asked through Python (for example, any calculations or equations that can be calculated).\n3. Wait for me to give the results.\n4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.\n\nAfter all the queries are run and you get the answer, put the answer in \\boxed{}.\n\nProblem:\nFor what negative value of $k$ is there exactly one solution to the system of equations \\begin{align*}\ny &= 2x^2 + kx + 6 \\\\\ny &= -x + 4?\n\\end{align*}\n### Assistant:"
    # solve inequality
    # prompt = "You are a helpful assistant.\n### Human: Got any creative ideas for a 10 year old’s birthday?\n### Assistant: Of course! Here are some creative ideas for a 10-year-old's birthday party:\n1. Treasure Hunt: Organize a treasure hunt in your backyard or nearby park. Create clues and riddles for the kids to solve, leading them to hidden treasures and surprises.\n2. Science Party: Plan a science-themed party where kids can engage in fun and interactive experiments. You can set up different stations with activities like making slime, erupting volcanoes, or creating simple chemical reactions.\n3. Outdoor Movie Night: Set up a backyard movie night with a projector and a large screen or white sheet. Create a cozy seating area with blankets and pillows, and serve popcorn and snacks while the kids enjoy a favorite movie under the stars.\n4. DIY Crafts Party: Arrange a craft party where kids can unleash their creativity. Provide a variety of craft supplies like beads, paints, and fabrics, and let them create their own unique masterpieces to take home as party favors.\n5. Sports Olympics: Host a mini Olympics event with various sports and games. Set up different stations for activities like sack races, relay races, basketball shooting, and obstacle courses. Give out medals or certificates to the participants.\n6. Cooking Party: Have a cooking-themed party where the kids can prepare their own mini pizzas, cupcakes, or cookies. Provide toppings, frosting, and decorating supplies, and let them get hands-on in the kitchen.\n7. Superhero Training Camp: Create a superhero-themed party where the kids can engage in fun training activities. Set up an obstacle course, have them design their own superhero capes or masks, and organize superhero-themed games and challenges.\n8. Outdoor Adventure: Plan an outdoor adventure party at a local park or nature reserve. Arrange activities like hiking, nature scavenger hunts, or a picnic with games. Encourage exploration and appreciation for the outdoors.\nRemember to tailor the activities to the birthday child's interests and preferences. Have a great celebration!\n### Human: Let's use Python to solve a math problem.\n\nQuery requirements:\nYou should always use the 'print' function for the output and use fractions/radical forms instead of decimals.\nYou can use packages like sympy to help you.\nYou must follow the formats below to write your code:\n```python\n# your code\n```\n\nFirst state the key idea to solve the problem. You may choose from three ways to solve the problem:\nCase 1: If the problem can be solved with Python code directly, please write a program to solve it. You can enumerate all possible arrangements if needed.\nCase 2: If the problem is mostly reasoning, you can solve it by yourself directly.\nCase 3: If the problem cannot be handled in the above two ways, please follow this process:\n1. Solve the problem step by step (do not over-divide the steps).\n2. Take out any queries that can be asked through Python (for example, any calculations or equations that can be calculated).\n3. Wait for me to give the results.\n4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.\n\nAfter all the queries are run and you get the answer, put the answer in \\boxed{}.\n\nProblem:\nFind all $x$ that satisfy the inequality $(2x+10)(x+3)<(3x+9)(x+8)$. Express your answer in interval notation.\n### Assistant:"

    # Generate predicted tokens
    with torch.inference_mode():
        # prompt = get_prompt(args.prompt, [], system_prompt=DEFAULT_SYSTEM_PROMPT)
        prompt = get_prompt(prompt, [], system_prompt=DEFAULT_SYSTEM_PROMPT)

        ###############   XPU Device  #################
        if args.device == 'xpu':
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to('xpu')
        else:
        ###############   CPU Device  #################
            input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # kwargs match with fastchat generate
        kwargs = {'top_p': 1.0, 'top_k': 50, 'temperature': 0.0}
        gen_config = GenerationConfig(**kwargs)

        # ipex model needs a warmup, then inference time can be accurate
        output = model.generate(input_ids,
                                generation_config=gen_config,
                                max_new_tokens=args.n_predict)
        
        # start inference
        print("Start Inferencing")
        st = time.time()
        # Here we use the is_greedy_gen_mode
        output = model.generate(input_ids,
                                generation_config=gen_config,
                                max_new_tokens=args.n_predict)
        
        
        ###############   XPU Device  #################
        if args.device == 'xpu':
            torch.xpu.synchronize()

        ###############   CPU Device  #################


        end = time.time()
        output = output.cpu()
        output_str = tokenizer.decode(output[0], skip_special_tokens=True)

        print(f'Inference time: {end-st} s')
        print('-'*20, 'Prompt', '-'*20)
        print(prompt)
        print('-'*20, 'Output', '-'*20)
        print(output_str)