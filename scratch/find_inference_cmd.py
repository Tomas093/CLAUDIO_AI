import json

log_path = r"C:\Users\Tomas\.gemini\antigravity\brain\1b174f0d-9f39-45d9-9c37-46b79304247a\.system_generated\logs\transcript.jsonl"

print("Searching for commands containing inference_sahi.py...")
with open(log_path, errors="ignore") as f:
    for line in f:
        if "inference_sahi.py" in line and "run_command" in line:
            try:
                data = json.loads(line)
                # Find tool_calls or commands
                tool_calls = data.get("tool_calls", [])
                for tc in tool_calls:
                    if tc.get("name") == "run_command":
                        cmd = tc.get("args", {}).get("CommandLine", "")
                        if "inference_sahi.py" in cmd:
                            print(f"- {cmd}")
            except Exception as e:
                pass
