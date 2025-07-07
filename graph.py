"""
This module defines the graph that orchestrates the financial analysis agents.

The graph is built using the LangGraph library and consists of nodes for each agent 
and a supervisor node that manages the overall workflow. The state of the graph is 
managed by a TypedDict, which ensures that the data flowing through the graph is 
well-structured.
"""

from typing import Annotated, List
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from financial_risk_analyzer.state import AgentState
from financial_risk_analyzer.agents import (
    market_risk_agent,
    credit_risk_agent,
    liquidity_risk_agent,
    operational_risk_agent,
    regulatory_risk_agent,
    supervisor_node,
)

# AgentState is now imported from state.py

# --- Agent Nodes ---
def market_risk_node(state: AgentState):
    report = market_risk_agent.invoke({"input": "Analyze market risk for UBS"})
    return {"market_risk_report": report['output']}

def credit_risk_node(state: AgentState):
    report = credit_risk_agent.invoke({"input": "Analyze credit risk for UBS"})
    return {"credit_risk_report": report['output']}

def liquidity_risk_node(state: AgentState):
    report = liquidity_risk_agent.invoke({"input": "Analyze liquidity risk for UBS"})
    return {"liquidity_risk_report": report['output']}

def operational_risk_node(state: AgentState):
    report = operational_risk_agent.invoke({"input": "Analyze operational risk for UBS"})
    return {"operational_risk_report": report['output']}

def regulatory_risk_node(state: AgentState):
    report = regulatory_risk_agent.invoke({"input": "Analyze regulatory and compliance risk for UBS"})
    return {"regulatory_risk_report": report['output']}

# supervisor_node is now imported from agents.py and does all processing


# --- Graph Definition ---
workflow = StateGraph(AgentState)

workflow.add_node("market_risk", market_risk_node)
workflow.add_node("credit_risk", credit_risk_node)
workflow.add_node("liquidity_risk", liquidity_risk_node)
workflow.add_node("operational_risk", operational_risk_node)
workflow.add_node("regulatory_risk", regulatory_risk_node)
workflow.add_node("supervisor", supervisor_node)  # This now uses the agents.py supervisor_node with postprocessing

# Set all risk nodes as entry points
workflow.set_entry_point("market_risk")
workflow.set_entry_point("credit_risk")
workflow.set_entry_point("liquidity_risk")
workflow.set_entry_point("operational_risk")
workflow.set_entry_point("regulatory_risk")
# Each risk node goes directly to supervisor
workflow.add_edge("market_risk", "supervisor")
workflow.add_edge("credit_risk", "supervisor")
workflow.add_edge("liquidity_risk", "supervisor")
workflow.add_edge("operational_risk", "supervisor")
workflow.add_edge("regulatory_risk", "supervisor")
workflow.add_edge("supervisor", END)

# Compile the graph
agent_graph = workflow.compile()
