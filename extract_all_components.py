with open('frontend/dist/assets/index-CkVcxXKw.js', 'r', encoding='utf-8') as f:
    text = f.read()

import re

components = ['Kn', 'uo', 'po', 'mo', 'ho', 'go', '_o', 'vo', 'B', 'H', 'fo', 'V']

for c in ['Kn=', 'uo=', 'po=', 'mo=', 'ho=', 'go=', '_o=', 'vo=', 'B=', 'H=', 'fo=', 'V=']:
    pos = text.find(c)
    if pos != -1:
        print(f"\n==================== COMPONENT: {c} ====================")
        snippet = text[pos:pos+3000]
        # find z.get, z.post etc in snippet
        print(snippet[:1800])
