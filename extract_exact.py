with open('frontend/dist/assets/index-CkVcxXKw.js', 'r', encoding='utf-8') as f:
    text = f.read()

for name in ['ho', 'Kn', 'mo', 'uo', 'po']:
    pos = text.find(f"{name}=")
    while pos != -1:
        if '()=>' in text[pos:pos+30] or 'function' in text[pos:pos+30]:
            print(f"\n==================== EXACT {name} (pos {pos}) ====================")
            print(text[pos:pos+2500])
            break
        pos = text.find(f"{name}=", pos + 1)
