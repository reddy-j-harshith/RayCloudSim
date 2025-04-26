import json
import random

# Load existing config
with open("config.json", "r") as f:
    config = json.load(f)

nodes = config["Nodes"]

# Determine how many nodes can be malicious (<30%)
max_malicious = int(len(nodes) * 0.3)
malicious_count = random.randint(1, max_malicious)

# Randomly select indices to mark as Malicious
malicious_indices = set(random.sample(range(len(nodes)), malicious_count))

# Assign types
for idx, node in enumerate(nodes):
    node["NodeType"] = "MaliciousNode" if idx in malicious_indices else "TrustNode"

# Save the modified config
with open("config_modified.json", "w") as f:
    json.dump(config, f, indent=4)
