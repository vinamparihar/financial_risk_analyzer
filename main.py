"""
This is the main entry point for the financial risk analysis application.

This script initializes the agent graph, sets the initial state, and runs the 
analysis. The final result, a comprehensive financial risk assessment, is then 
printed to the console.
"""

import os
from dotenv import load_dotenv
from financial_risk_analyzer.graph import agent_graph

# Load environment variables from .env file
load_dotenv()

def main():
    """Main function to run the financial risk analysis."""
    print("Starting financial risk analysis for UBS Group AG...")

    # Define the initial state for the graph
    initial_state = {
        "market_risk_report": "",
        "credit_risk_report": "",
        "liquidity_risk_report": "",
        "operational_risk_report": "",
        "regulatory_risk_report": "",
        "final_report": "",
    }

    # Stream the execution of the graph
    for output in agent_graph.stream(initial_state):
        for key, value in output.items():
            print(f"--- {key.replace('_', ' ').title()} ---")
            print(value)
            print("\n")

    print("\n--- End of Analysis ---")

if __name__ == "__main__":
    # Set your API keys in a .env file
    main()

