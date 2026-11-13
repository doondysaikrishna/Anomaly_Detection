with open('frontend/dist/assets/index-CkVcxXKw.js', 'r', encoding='utf-8') as f:
    text = f.read()

pos_mo = text.find('mo=()=>')
print(text[pos_mo:pos_mo+4000])
