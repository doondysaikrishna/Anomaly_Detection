with open('frontend/dist/assets/index-CkVcxXKw.js', 'r', encoding='utf-8') as f:
    text = f.read()

import re

# Find all occurrences of z.get and z.post with context
matches = re.finditer(r'z\.(get|post|put|delete)\(([^)]+)\)', text)
endpoints = []
for m in matches:
    start = max(0, m.start() - 200)
    end = min(len(text), m.end() + 400)
    endpoints.append((m.group(1), m.group(2), text[start:end]))

with open('endpoints_analysis.txt', 'w', encoding='utf-8') as out:
    for method, arg, ctx in endpoints:
        out.write(f"=== {method.upper()} {arg} ===\n{ctx}\n\n" + "="*60 + "\n\n")

print(f"Extracted {len(endpoints)} endpoint calls to endpoints_analysis.txt")
