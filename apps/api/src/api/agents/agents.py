from langsmith import traceable, get_current_run_tree

from langchain_core.messages import convert_to_openai_messages, AIMessage
from openai import OpenAI
import instructor

from api.agents.utils.prompt_management import prompt_template_config
from api.agents.utils.utils import format_ai_message
from pydantic import BaseModel, Field
from typing import List

class ToolCall(BaseModel):
    name: str
    arguments: dict

### QnA Agent Response Model

class RAGUsedContext(BaseModel):
    id: str = Field(description="The ID of the item used to answer the question")
    description: str = Field(description="Short description of the item used to answer the question")
    price: float | None = None
    
class ProductQAAgentResponse(BaseModel):
    answer: str = Field(description="Answer to the question.")
    references: list[RAGUsedContext] = Field(description="List of items used to answer the question.")
    final_answer: bool = False
    tool_calls: List[ToolCall] = []


### Shopping Cart Agent Response Model

class ShoppingCartAgentResponse(BaseModel):
    answer: str = Field(description="Answer to the question.")
    final_answer: bool = False
    tool_calls: List[ToolCall] = []

### Warehosue Manager Agent Response Model

class WarehouseManagerAgentResponse(BaseModel):
    answer: str = Field(description="Answer to the question.")
    final_answer: bool = False
    tool_calls: List[ToolCall] = []

### Coordinator Response Model

class Delegation(BaseModel):
    agent: str
    task: str

class CoordinatorAgentResponse(BaseModel):
    next_agent: str
    plan: List[Delegation]
    final_answer: bool = False
    answer: str = ""

def _safe_trace_id() -> str:
    current_run = get_current_run_tree()
    if not current_run:
        return ""
    # prefer trace_id if present, otherwise fall back to run id
    return str(getattr(current_run, "trace_id", "") or getattr(current_run, "id", "") or "")

### Product QA Agent

@traceable(
    name="product_qa_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"}
)
def product_qa_agent(state) -> dict:

   template = prompt_template_config("api/agents/prompts/product_qa_agent.yaml", "product_qa_agent")
   
   prompt = template.render(
      available_tools=state.product_qa_agent.available_tools
   )

   messages = state.messages

   conversation = []

   for message in messages:
        conversation.append(convert_to_openai_messages(message))

   client = instructor.from_openai(OpenAI())

   response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1",
        response_model=ProductQAAgentResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0.5,
   )

   current_run = get_current_run_tree()

   if current_run:
       current_run.metadata["usage_metadata"] = {
           "input_tokens": raw_response.usage.prompt_tokens,
           "output_tokens": raw_response.usage.completion_tokens,
           "total_tokens": raw_response.usage.total_tokens
       }

   trace_id = _safe_trace_id()

   ai_message = format_ai_message(response)

   return {
        "messages": [ai_message],
        "product_qa_agent": {
            "tool_calls": [tool_call.model_dump() for tool_call in response.tool_calls],
            "iteration": state.product_qa_agent.iteration + 1,
            "final_answer": response.final_answer,
            "available_tools": state.product_qa_agent.available_tools
        },
        "answer": response.answer,
        "references": response.references
   }


### Shopping Cart Agent

@traceable(
    name="shopping_cart_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"}
)
def shopping_cart_agent(state) -> dict:

   template = prompt_template_config("api/agents/prompts/shopping_cart_agent.yaml", "shopping_cart_agent") 

   prompt = template.render(
      available_tools=state.shopping_cart_agent.available_tools,
      user_id=state.user_id,
      cart_id=state.cart_id
   )

   messages = state.messages

   conversation = []

   for message in messages:
        conversation.append(convert_to_openai_messages(message))

   client = instructor.from_openai(OpenAI())

   response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1",
        response_model=ShoppingCartAgentResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0.5,
   )

   current_run = get_current_run_tree()

   if current_run:
       current_run.metadata["usage_metadata"] = {
           "input_tokens": raw_response.usage.prompt_tokens,
           "output_tokens": raw_response.usage.completion_tokens,
           "total_tokens": raw_response.usage.total_tokens
       }

   trace_id = _safe_trace_id()

   ai_message = format_ai_message(response)

   return {
      "messages": [ai_message],
      "shopping_cart_agent": {
         "iteration": state.shopping_cart_agent.iteration + 1,
         "final_answer": response.final_answer,
         "tool_calls": [tool_call.model_dump() for tool_call in response.tool_calls],
         "available_tools": state.shopping_cart_agent.available_tools
      },
      "answer": response.answer
   }

### Warehouse Manager Agent

@traceable(
    name="warehouse_manager_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"}
)
def warehouse_manager_agent(state) -> dict:

   template = prompt_template_config("api/agents/prompts/warehouse_manager_agent.yaml", "warehouse_manager_agent") 

   prompt = template.render(
      available_tools=state.warehouse_manager_agent.available_tools,
      user_id=state.user_id,
      cart_id=state.cart_id
   )

   messages = state.messages

   conversation = []

   for message in messages:
        conversation.append(convert_to_openai_messages(message))

   client = instructor.from_openai(OpenAI())

   response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1",
        response_model=WarehouseManagerAgentResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0.5,
   )

   current_run = get_current_run_tree()

   if current_run:
       current_run.metadata["usage_metadata"] = {
           "input_tokens": raw_response.usage.prompt_tokens,
           "output_tokens": raw_response.usage.completion_tokens,
           "total_tokens": raw_response.usage.total_tokens
       }

   trace_id = _safe_trace_id()

   ai_message = format_ai_message(response)

   return {
      "messages": [ai_message],
      "warehouse_manager_agent": {
         "iteration": state.warehouse_manager_agent.iteration + 1,
         "final_answer": response.final_answer,
         "tool_calls": [tool_call.model_dump() for tool_call in response.tool_calls],
         "available_tools": state.warehouse_manager_agent.available_tools
      },
      "answer": response.answer
   }

### Intent Router Agent Node


@traceable(
    name="coordinator_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"}
)
def coordinator_agent(state):

   template = prompt_template_config("api/agents/prompts/coordinator_agent.yaml", "coordinator_agent") 
   
   prompt = template.render()

   messages = state.messages

   conversation = []

   for message in messages:
        conversation.append(convert_to_openai_messages(message))

   client = instructor.from_openai(OpenAI())

   response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1",
        response_model=CoordinatorAgentResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0.5,
   )

   current_run = get_current_run_tree()

   if current_run:
       current_run.metadata["usage_metadata"] = {
           "input_tokens": raw_response.usage.prompt_tokens,
           "output_tokens": raw_response.usage.completion_tokens,
           "total_tokens": raw_response.usage.total_tokens
       }
   
   trace_id = _safe_trace_id()


   if response.final_answer:
      ai_message = [AIMessage(
         content=response.answer,
      )]
   else:
      ai_message = []

   return {
      "messages": ai_message,
      "answer": response.answer,
      "coordinator_agent": {
         "iteration": state.coordinator_agent.iteration + 1,
         "final_answer": response.final_answer,
         "next_agent": response.next_agent,
         "plan": [data.model_dump() for data in response.plan]
      },
      "trace_id": trace_id
   }