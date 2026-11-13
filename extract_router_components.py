with open('frontend/dist/assets/index-CkVcxXKw.js', 'r', encoding='utf-8') as f:
    text = f.read()

# Search for the function definitions towards the end of the bundle
import re

endpoints_info = []

for match in re.finditer(r'([a-zA-Z0-9_$]+)=function\(\)|([a-zA-Z0-9_$]+)=\(\)=>[{]', text):
    name = match.group(1) or match.group(2)
    if name in ['uo', 'Kn', 'po', 'mo', 'ho', 'go', '_o', 'vo', 'B', 'H', 'fo', 'V', 'yo']:
        pos = match.start()
        # Find next function or 3000 chars
        chunk = text[pos:pos+3500]
        print(f"\n==================== COMPONENT {name} (pos {pos}) ====================")
        print(chunk[:2500])
