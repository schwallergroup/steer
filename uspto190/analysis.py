
import os
import json

nroutes = {}
for file in os.listdir("uspto190/chimera"):
    with open(f"uspto190/chimera/{file}", "r") as f:
        if 'error' not in file:
            data = json.load(f)
            nroutes[file] = len(data)

print(json.dumps(nroutes, indent=4))