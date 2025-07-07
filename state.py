from typing import TypedDict

class AgentState(TypedDict):
    market_risk_report: str
    credit_risk_report: str
    liquidity_risk_report: str
    operational_risk_report: str
    regulatory_risk_report: str
    final_report: str
