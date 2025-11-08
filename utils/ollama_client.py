import shutil
import subprocess

def ollama_query(prompt: str, model: str = "llama3.2") -> str:
    if not shutil.which("ollama"):
        return f"[SIMULATED OLLAMA] {prompt[:200]}..."
    try:
        proc = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode(),
            capture_output=True,
            timeout=600
        )
        return proc.stdout.decode().strip() or proc.stderr.decode().strip()
    except Exception as e:
        return f"[OLLAMA ERROR] {e}"