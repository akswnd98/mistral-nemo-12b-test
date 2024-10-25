from typing import List, Tuple, Any
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_community.llms.llamacpp import LlamaCpp
from transformers import AutoTokenizer, LlamaTokenizer, PreTrainedTokenizer
from json import dumps, loads
from langchain.agents.agent import AgentOutputParser
from tools import get_current_weather, send_email, search_web
import string
import random

def generate_history_passthrough_function ():
  def history_passthrough_function (inputs):
    # format
    # {'name': 'get_current_weather', 'arguments': {'location': 'Seoul, KR', 'format': 'celsius'}}
    intermediate_steps: List[Tuple[AgentAction, str]] = inputs['intermediate_steps']
    input: str = inputs['input']
    ret = [
      {'role': 'user', 'content': input}
    ]
    for intermediate_step in intermediate_steps:
      tool_call = {'name': intermediate_step[0].tool, 'arguments': intermediate_step[0].tool_input}
      tool_call_id = ''.join(random.choices(string.ascii_lowercase + string.ascii_uppercase + string.digits, k=9))
      ret += [
        {'role': 'assistant', 'tool_calls': [{'id': tool_call_id, 'type': 'function', 'function': tool_call}]},
        {'role': 'tool', 'tool_call_id': tool_call_id, 'name': intermediate_step[0].tool, 'content': intermediate_step[1]}
      ]

    return ret

  return history_passthrough_function

def generate_llm_function (tokenizer: PreTrainedTokenizer, llm: LlamaCpp):
  tools = [get_current_weather.func, send_email.func, search_web.func]

  def llm_function (inputs):
    history: List[dict[str, Any]] = inputs['history']
    llm_inputs = tokenizer.apply_chat_template(
      conversation=history,
      tools=tools,
      add_generation_prompt=True,
      return_dict=True,
      return_tensors="pt",
    )
    ret = llm.invoke(tokenizer.decode(llm_inputs['input_ids'][0]))
    try:
      ret = loads(ret)
    except Exception as e:
      return ret

    return dumps(ret[0])

  return llm_function

class MistralAgentOutputParser (AgentOutputParser):
  def parse (self, text: str) -> AgentAction | AgentFinish:
    try:
      jsonified = loads(text)
    except Exception as e:
      return AgentFinish({'output': text}, text)

    # if jsonified['name'] == 'finish':
      # return AgentFinish({'output': jsonified['arguments']['final_answer']}, jsonified['arguments']['final_answer'])

    return AgentAction(jsonified['name'], jsonified['arguments'], 'no log')
