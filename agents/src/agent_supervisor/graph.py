from typing import Annotated, TypedDict, Union, Literal, Dict
from langchain_core.messages import BaseMessage
import operator
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

from src.agent_supervisor.prompts import (
    SUPERVISOR_CHAT_PROMPT_TEMPLATE,
    GENERATE_CLARIFICATION_PROMPT_TEMPLATE,
    GENERATE_RESPONSE_PROMPT_TEMPLATE,
)
from src.agent_supervisor.config import Configuration


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], operator.add]
    supervisor_decision: Union[None, Literal["respond", "clarify"]]
    intermediate_steps: Annotated[list[dict], operator.add]


def supervisor(state: AgentState) -> Dict:
    """Supervisor agent that decides whether to respond or clarify the user's message.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        Dict: The next state of the agent.
    """

    # Create LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Parse supervisor response
    parser = JsonOutputParser()

    # Get supervisor decision
    chain = SUPERVISOR_CHAT_PROMPT_TEMPLATE | llm | parser

    response = chain.invoke({"messages": state["messages"]})

    # Update state with supervisor decision
    return {
        "supervisor_decision": response["decision"],
        "intermediate_steps": [response],
    }


def generate_response(state: AgentState, config: Configuration) -> Dict:
    """Generate a response to the user's message.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        Dict: The next state of the agent.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = GENERATE_RESPONSE_PROMPT_TEMPLATE | llm
    response = chain.invoke(
        {"messages": state["messages"], "name": config["configurable"]["name"]}
    )
    return {"messages": [response]}


def generate_clarification(state: AgentState) -> Dict:
    """Generate a clarification for the user's message.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        Dict: The next state of the agent.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = GENERATE_CLARIFICATION_PROMPT_TEMPLATE | llm
    response = chain.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def supervisor_router(state: AgentState) -> str:
    """Router for the supervisor agent.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        str: The next state of the agent.
    """
    # Define a router function that returns the next node name
    if state["supervisor_decision"] == "respond":
        return "generate_response"
    elif state["supervisor_decision"] == "clarify":
        return "generate_clarification"
    else:
        raise ValueError("Invalid supervisor decision")


builder = StateGraph(AgentState, config_schema=Configuration)

builder.add_node("supervisor", supervisor)
builder.add_node("generate_response", generate_response)
builder.add_node("generate_clarification", generate_clarification)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    supervisor_router,
    path_map=["generate_response", "generate_clarification"],
)
builder.add_edge("generate_response", END)
builder.add_edge("generate_clarification", END)

graph = builder.compile()
