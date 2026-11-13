import re

with open('frontend/dist/assets/index-CkVcxXKw.js', 'r', encoding='utf-8') as f:
    text = f.read()

matches = re.findall(r'(\.[a-z]+)\([`\'"](/[^`\'"]+)[`\'"]', text)
print("=== Matches ===")
for m in sorted(set(matches)):
    print(m)

print("\n=== All /api or relative endpoints ===")
matches2 = re.findall(r'[`\'"](/[a-zA-Z0-9_\-\/]+)[`\'"]', text)
for m in sorted(set(matches2)):
    if any(k in m for k in ['auth', 'detect', 'seg', 'analy', 'dash', 'hist', 'report', 'model', 'stat', 'setting', 'user', 'export', 'alert', 'overview']):
        print(m)
