with open('frontend/dist/assets/index-CkVcxXKw.js', 'r', encoding='utf-8') as f:
    text = f.read()

import re

# Find all component definitions in the range pos > 280000
for name in ['uo', 'Kn', 'po', 'mo', 'ho', 'go', '_o', 'vo', 'B', 'H', 'fo', 'V']:
    pos = text.rfind(f"{name}=")
    if pos != -1:
        print(f"\n==================== {name} (pos {pos}) ====================")
        print(text[pos:pos+3500])
