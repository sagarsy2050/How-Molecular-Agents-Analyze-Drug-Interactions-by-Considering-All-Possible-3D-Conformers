from utils.ollama_client import ollama_query
import json

with open("config/agents.json") as f:
    AGENTS = json.load(f)["agents"]

def run_agent(name: str, context: str, use_ollama: bool, model: str):
    print(f"\n{'='*30} AGENT: {name} {'='*30}")
    prompt = f"""You are {name}: Atomic-level drug discovery genius.
Analyze this 3D conformer with charges, geometry, dipole, PMI:
1. Key atomic insights
2. Binding implications
3. SAR trends
4. Optimization ideas

Context: {context}"""
    result = ollama_query(prompt, model) if use_ollama else f"[SIMULATED] {name}: Breakthrough atomic insight."
    print(result[:700] + ("..." if len(result) > 700 else ""))
    return result