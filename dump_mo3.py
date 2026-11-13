with open('frontend/dist/assets/index-CkVcxXKw.js', 'r', encoding='utf-8') as f:
    text = f.read()

pos_mo = text.find('mo=()=>')
print(text[pos_mo+3000:pos_mo+6000])
