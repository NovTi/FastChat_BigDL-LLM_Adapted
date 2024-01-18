## Running AutoGen with Local LLM


#### 1. Create New Folder
```bash
# make a new folder for autogen testing
mkdir autogen_test
cd autogen_test
```


#### 2. Setup FastChat and AutoGen Environment
```bash
cd autogen_test
# operations are in the autogen_test folder
# setup FastChat environment
git clone https://github.com/NovTi/FastChat_BigDL-LLM_Adapted.git FastChat # clone the bigdl adapted version 
cd FastChat
pip3 install --upgrade pip  # enable PEP 660 support
pip3 install -e ".[model_worker,webui]"

# setup AutoGen environment
pip install pyautogen==0.1.14
```

#### 3. Build FastChat OpenAI-Compatible RESTful API
Open 4 terminals

**Terminal 1: Launch the controller**

```bash
cd autogen_test
cd FastChat  # go to the cloned FastChat folder in autogen_test folder
python -m fastchat.serve.controller
```

**Terminal 2: Launch the workers**

```bash
cd autogen_test
cd FastChat  # go to the cloned FastChat folder in autogen_test folder
# original one | device choice: [xpu, cpu]
python -m fastchat.serve.model_worker --model-path YOUR_MODEL_PATH --device YOUR_DEVICE
# bigdl load 4bit | device choice: [xpu, cpu]
python -m fastchat.serve.model_worker --model-path YOUR_MODEL_PATH --device YOUR_DEVICE --bigdl_load --load_in_4bit
```

**Terminal 3: Launch the server**

```bash
cd autogen_test
cd FastChat  # go to the cloned FastChat folder in autogen_test
python -m fastchat.serve.openai_api_server --host localhost --port 8000
```

**Terminal 4: Testing with AutoGen**

```bash
cd autogen_test

# copy the test scripts from the cloned FastChat
cp -r FastChat/autogen_test_scripts .

# go to the test scripts folder
cd autogen_test_scripts

# test autogen examples in the autogen_test_scripts folder
# Example_Files: [math_chat_solve_equations.py, math_chat_solve_inequality.py, teaching.py, generate.py]
python Example_Files

# generate scripts | device choice: [xpu, cpu]
# the prompts are in the line 74-78 in the generate file
python -m generate --repo-id-or-model-path YOUR_MODEL_PATH --device YOUR_DEVICE
```

## Notices

#### Deal with 0.1.14 AutoGen cache folder
AutoGen automatically stores the cache every time it gets an output. Considering there are four settings for each task (ipex xpu, ipen cpu, bigdl 4bit xpu, bigdl 4bit cpu), you can choose to add `use_cache=False` at line 203 of the PYTHON_PATH/site-packages/autogen/oai/completion.py. This exempts deleting the cache file every time before testing different settings.

#### Get Prompt
Prompt can be got from file `FastChat/fastchat/serve/inference.py` line75 `prompt = params["prompt"]`. If you want to print the prompt, the results will be printed in the Terminal 2, it will not be shown in the Terminal 4 that used for running python files.

#### File Illustration
- The [math_chat_solve_equations.py](math_chat_solve_equations.py) is the AutoGen solve math equation example.
- The [math_chat_solve_inequality.py](math_chat_solve_inequality.py) is the AutoGen solve math inequality example.
- The [teaching.py](teaching.py) is the AutoGen teaching how to use Arxiv example.
- The [generate.py](generate.py) provides prompts for math solve equations and solve inequalities examples. They are in the line 74-78 of the file.

#### Verified Models
These files were tested with `Llama-2-7b-chat-hf` model. If you wish to test with other models, please change the respective model name in the [Math Solve Equations](math_chat_solve_equations.py)(line 36 config_list), [Math Solve Inequalities](math_chat_solve_inequality.py)(line 36 config_list), and [Teaching](teaching.py)(line 35 config_list) files.