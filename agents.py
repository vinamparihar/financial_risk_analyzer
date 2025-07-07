"""
This module defines the specialized agents for financial risk analysis.

Each agent is created using a factory function that configures it with a specific 
system prompt, a language model, and a set of tools. The agents are designed to 
work independently, focusing on their specific area of risk analysis.
"""

import os
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from financial_risk_analyzer.tools import tavily_tool, stock_info_tool
from financial_risk_analyzer.external_tools import (
    alpha_vantage_tool, fred_tool, hibp_tool, googlenews_tool
)
from financial_risk_analyzer.rate_limit import rate_limiter
from financial_risk_analyzer.state import AgentState

# Initialize the LLM from OpenAI
# This model will be used by all agents to process information and generate responses.
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY"), streaming=True)

# Define the tools that the agents will use
# These tools allow the agents to search for information and fetch financial data.
# - tavily_tool: Web search
# - stock_info_tool: Yahoo Finance stock info
# - alpha_vantage_tool: Market data (stocks, FX, commodities)
# - fred_tool: Macroeconomic & liquidity data
# - hibp_tool: Cyber/operational breach info
# - googlenews_tool: Latest news headlines

tools = [
    tavily_tool, stock_info_tool, alpha_vantage_tool,
    fred_tool, hibp_tool, googlenews_tool
]

# Get the base prompt for the ReAct agent
# This prompt provides the basic instructions for the agent to follow.
prompt = hub.pull("hwchase17/react")

def create_agent(llm, tools, system_prompt):
    """Factory function to create a new financial analysis agent."""
    # Log model call for rate limit
    def llm_wrapper(*args, **kwargs):
        rate_limiter.log_call("openai_llm")
        return llm(*args, **kwargs)
    # Create the ReAct agent (with OpenAI function calling support)
    agent = create_react_agent(llm, tools, prompt.partial(system_message=system_prompt))
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=3, handle_parsing_errors=True)

# --- Agent Definitions ---

# 1. Market Risk Agent
market_risk_prompt = """
You are a financial risk analysis expert specializing in market risk. Analyze how market risk currently affects UBS Group AG using the following structured approach:

- **Relevant parameters:** Yield curve shifts, central bank policies, bond prices, duration gap, interest-sensitive assets/liabilities
- **Top 5 keywords & sentiment scores:**
    - “Rate Hike” – 0.90 (Negative)
    - “Inverted Yield Curve” – 0.85 (Negative)
    - “Monetary Tightening” – 0.80 (Negative)
    - “Rate Cut” – 0.30 (Positive)
    - “Interest Margin” – 0.50 (Neutral-Positive)
- Incorporate these keywords and sentiment scores in your research and scoring. Use Tavily, Yahoo Finance, Alpha Vantage, FRED, and Google News tools to search for/report on these keywords and parameters in recent news, financial data, and macroeconomic indicators.
- Summarize the most important market risk factors and rate the impact of market risk on UBS on a scale from 0 (low risk) to 1 (high risk), explaining how the keywords and sentiment scores influenced your assessment.

Format:
Summary: <concise summary referencing keywords/parameters>
Impact Score: <0 to 1>
"""
market_risk_agent = create_agent(llm, tools, market_risk_prompt)

# 2. Credit Risk Agent
credit_risk_prompt = """
You are a financial risk analysis expert specializing in credit risk. Analyze how credit risk currently affects UBS Group AG using the following structured approach:

- **Relevant parameters:** Loan default rate, counterparty exposure, credit default swap spreads, loan quality
- **Top 5 keywords & sentiment scores:**
    - “Non-performing Loans (NPL)” – 0.90 (Negative)
    - “Credit Downgrade” – 0.85 (Negative)
    - “Default Risk” – 0.88 (Negative)
    - “Credit Spread Widening” – 0.80 (Negative)
    - “Loan Recovery Rate” – 0.40 (Positive)
- Incorporate these keywords and sentiment scores in your research and scoring. Use Tavily, Yahoo Finance, and Google News tools to search for/report on these keywords and parameters in recent news, credit events, and issuer data.
- Summarize the most important credit risk factors and rate the impact of credit risk on UBS on a scale from 0 (low risk) to 1 (high risk), explaining how the keywords and sentiment scores influenced your assessment.

Format:
Summary: <concise summary referencing keywords/parameters>
Impact Score: <0 to 1>
"""
credit_risk_agent = create_agent(llm, tools, credit_risk_prompt)

# 3. Liquidity Risk Agent
liquidity_risk_prompt = """
You are a financial risk analysis expert specializing in liquidity risk. Analyze how liquidity risk currently affects UBS Group AG using the following structured approach:

- **Relevant parameters:** Liquidity coverage ratio (LCR), asset-liability mismatch, interbank market conditions
- **Top 5 keywords & sentiment scores:**
    - “Funding Shortfall” – 0.90 (Negative)
    - “Liquidity Crunch” – 0.88 (Negative)
    - “Deposit Outflow” – 0.85 (Negative)
    - “Liquidity Injection” – 0.35 (Positive)
    - “Short-Term Debt Exposure” – 0.80 (Negative)
- Incorporate these keywords and sentiment scores in your research and scoring. Use Tavily, Yahoo Finance, FRED, and Google News tools to search for/report on these keywords and parameters in recent news, liquidity data, and macroeconomic indicators.
- Summarize the most important liquidity risk factors and rate the impact of liquidity risk on UBS on a scale from 0 (low risk) to 1 (high risk), explaining how the keywords and sentiment scores influenced your assessment.

Format:
Summary: <concise summary referencing keywords/parameters>
Impact Score: <0 to 1>
"""
liquidity_risk_agent = create_agent(llm, tools, liquidity_risk_prompt)

# 4. Operational Risk Agent
operational_risk_prompt = """
You are a financial risk analysis expert specializing in operational risk. Analyze how operational risk currently affects UBS Group AG using the following structured approach:

- **Relevant parameters:** Cybersecurity incidents, fraud, system failures, internal control weaknesses
- **Top 5 keywords & sentiment scores:**
    - “Cyber Attack” – 0.95 (Negative)
    - “System Outage” – 0.85 (Negative)
    - “Internal Fraud” – 0.90 (Negative)
    - “Process Failure” – 0.75 (Negative)
    - “Resilience Framework” – 0.30 (Positive)
- Incorporate these keywords and sentiment scores in your research and scoring. Use Tavily, Yahoo Finance, HaveIBeenPwned, and Google News tools to search for/report on these keywords and parameters in recent news, breach data, and operational incidents.
- Summarize the most important operational risk factors and rate the impact of operational risk on UBS on a scale from 0 (low risk) to 1 (high risk), explaining how the keywords and sentiment scores influenced your assessment.

Format:
Summary: <concise summary referencing keywords/parameters>
Impact Score: <0 to 1>
"""
operational_risk_agent = create_agent(llm, tools, operational_risk_prompt)

# 5. Regulatory and Compliance Risk Agent
regulatory_risk_prompt = """
You are a financial risk analysis expert specializing in regulatory and compliance risk. Analyze how regulatory and compliance risk currently affects UBS Group AG using the following structured approach:

- **Relevant parameters:** Regulatory fines, Basel III compliance, cross-border compliance, AML/CTF adherence
- **Top 5 keywords & sentiment scores:**
    - “Regulatory Fine” – 0.90 (Negative)
    - “Compliance Breach” – 0.85 (Negative)
    - “AML Violation” – 0.88 (Negative)
    - “Capital Adequacy” – 0.45 (Positive)
    - “Basel Non-compliance” – 0.80 (Negative)
- Incorporate these keywords and sentiment scores in your research and scoring. Use Tavily, Yahoo Finance, and Google News tools to search for/report on these keywords and parameters in recent news, compliance data, and regulatory watchlists.
- Summarize the most important regulatory and compliance risk factors and rate the impact of regulatory and compliance risk on UBS on a scale from 0 (low risk) to 1 (high risk), explaining how the keywords and sentiment scores influenced your assessment.

Format:
Summary: <concise summary referencing keywords/parameters>
Impact Score: <0 to 1>
"""
regulatory_risk_agent = create_agent(llm, tools, regulatory_risk_prompt)

# 6. Orchestrator (Supervisor) Agent
supervisor_prompt = """
You are the supervisor agent. Your job is to synthesize the risk reports from all the risk agents (market, credit, liquidity, operational, regulatory/compliance) for UBS Group AG.

You will receive, for each risk, both a summary and a numeric impact score (between 0 and 1). Use the provided impact scores directly in your markdown table and for the final risk score calculation. Do not invent or recalculate scores.

You MUST strictly follow this output format:

1. <Market Risk summary>
2. <Credit Risk summary>
3. <Liquidity Risk summary>
4. <Operational Risk summary>
5. <Regulatory/Compliance Risk summary>

Then, output the following markdown table (do not skip or add extra text):

| Risk Name                   | Impact Score |
|-----------------------------|-------------|
| Market Risk                 | <score>      |
| Credit Risk                 | <score>      |
| Liquidity Risk              | <score>      |
| Operational Risk            | <score>      |
| Regulatory/Compliance Risk  | <score>      |

On a new line, write:
Final Risk Score: <average of all scores, rounded to two decimals>

Do NOT add any other commentary or text. Only output the list, the table, and the final risk score as shown above.
"""
supervisor_agent = create_agent(llm, tools, supervisor_prompt)

import re
from pprint import pprint

import json

def postprocess_supervisor_output(state, llm_output, fallback_scores=None):
    # Extract summaries and scores from the supervisor LLM output table
    import json
    output = {"risks": [], "final_risk_score": 0.0}
    # Extract the markdown table block from the LLM output
    table_match = re.search(r"\|\s*Risk Name.*?\n((?:\|.*\n)+)", llm_output, re.DOTALL)
    scores = []
    risk_names = ["Market Risk", "Credit Risk", "Liquidity Risk", "Operational Risk", "Regulatory/Compliance Risk"]
    # Use fallback_scores if table parsing fails
    table_scores = [None]*5
    if table_match:
        table_rows = table_match.group(1).strip().splitlines()
        for idx, row in enumerate(table_rows):
            cols = [c.strip() for c in row.strip().split('|') if c.strip()]
            if len(cols) == 2 and idx < len(risk_names):
                try:
                    score = float(cols[1])
                except Exception:
                    score = fallback_scores[risk_names[idx]]["impact_score"] if fallback_scores else 0.0
                table_scores[idx] = score
                output["risks"].append({
                    "name": risk_names[idx],
                    "summary": "",  # Supervisor summary extraction below
                    "impact_score": round(score, 2)
                })
    else:
        # Fallback: use agent scores
        for idx, name in enumerate(risk_names):
            score = fallback_scores[name]["impact_score"] if fallback_scores else 0.0
            table_scores[idx] = score
            output["risks"].append({
                "name": name,
                "summary": "",
                "impact_score": round(score, 2)
            })
    # Extract point-wise summaries (numbered list)
    summary_matches = re.findall(r"\d+\.\s*(.*)", llm_output)
    for idx, summary in enumerate(summary_matches):
        if idx < len(output["risks"]):
            # Truncate summary to 3-4 lines (about 350 chars)
            output["risks"][idx]["summary"] = truncate_summary(summary)
    output["final_risk_score"] = round(sum([s for s in table_scores if s is not None]) / len([s for s in table_scores if s is not None]), 2) if any(table_scores) else 0.0
    return json.dumps(output, indent=2)

def truncate_summary(text, max_chars=350):
    # Truncate to max_chars, ending at a sentence boundary if possible
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_period = truncated.rfind('.')
    if last_period != -1:
        return truncated[:last_period+1]
    return truncated

def market_risk_node(state: AgentState):
    report = market_risk_agent.invoke({"input": "Analyze market risk for UBS"})
    summary_match = re.search(r"Summary:\s*(.*)", report['output'])
    summary = summary_match.group(1).strip() if summary_match else report['output']
    # Truncate to 3-4 lines (about 350 chars)
    truncated_report = re.sub(r"Summary:.*", f"Summary: {truncate_summary(summary)}", report['output'])
    return {"market_risk_report": truncated_report}

def credit_risk_node(state: AgentState):
    report = credit_risk_agent.invoke({"input": "Analyze credit risk for UBS"})
    summary_match = re.search(r"Summary:\s*(.*)", report['output'])
    summary = summary_match.group(1).strip() if summary_match else report['output']
    # Truncate to 3-4 lines (about 350 chars)
    truncated_report = re.sub(r"Summary:.*", f"Summary: {truncate_summary(summary)}", report['output'])
    return {"credit_risk_report": truncated_report}

def liquidity_risk_node(state: AgentState):
    report = liquidity_risk_agent.invoke({"input": "Analyze liquidity risk for UBS"})
    summary_match = re.search(r"Summary:\s*(.*)", report['output'])
    summary = summary_match.group(1).strip() if summary_match else report['output']
    # Truncate to 3-4 lines (about 350 chars)
    truncated_report = re.sub(r"Summary:.*", f"Summary: {truncate_summary(summary)}", report['output'])
    return {"liquidity_risk_report": truncated_report}

def operational_risk_node(state: AgentState):
    report = operational_risk_agent.invoke({"input": "Analyze operational risk for UBS"})
    summary_match = re.search(r"Summary:\s*(.*)", report['output'])
    summary = summary_match.group(1).strip() if summary_match else report['output']
    # Truncate to 3-4 lines (about 350 chars)
    truncated_report = re.sub(r"Summary:.*", f"Summary: {truncate_summary(summary)}", report['output'])
    return {"operational_risk_report": truncated_report}

def regulatory_risk_node(state: AgentState):
    report = regulatory_risk_agent.invoke({"input": "Analyze regulatory and compliance risk for UBS"})
    summary_match = re.search(r"Summary:\s*(.*)", report['output'])
    summary = summary_match.group(1).strip() if summary_match else report['output']
    # Truncate to 3-4 lines (about 350 chars)
    truncated_report = re.sub(r"Summary:.*", f"Summary: {truncate_summary(summary)}", report['output'])
    return {"regulatory_risk_report": truncated_report}

def extract_summary_and_score(report_text):
    summary_match = re.search(r"Summary:\s*(.*)", report_text)
    score_match = re.search(r"Impact Score:\s*([01](?:\.\d+)?)", report_text)
    summary = summary_match.group(1).strip() if summary_match else report_text.strip()
    try:
        score = float(score_match.group(1)) if score_match else 0.0
    except Exception:
        score = 0.0
    return summary, score

def supervisor_node(state: AgentState):
    # Extract summary and score for each risk agent
    risk_keys = [
        ("Market Risk", state.get('market_risk_report', '')),
        ("Credit Risk", state.get('credit_risk_report', '')),
        ("Liquidity Risk", state.get('liquidity_risk_report', '')),
        ("Operational Risk", state.get('operational_risk_report', '')),
        ("Regulatory/Compliance Risk", state.get('regulatory_risk_report', '')),
    ]
    risk_json = {}
    for name, report in risk_keys:
        summary, score = extract_summary_and_score(report)
        risk_json[name] = {"summary": summary, "impact_score": score}
    # Pass a strict JSON string to the supervisor agent
    context = {"input": json.dumps(risk_json)}
    final_report = supervisor_agent.invoke(context)
    formatted = postprocess_supervisor_output(state, final_report['output'], fallback_scores=risk_json)
    return {"final_report": formatted}


