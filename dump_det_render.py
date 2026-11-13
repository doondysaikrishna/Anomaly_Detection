with open('frontend/dist/assets/index-CkVcxXKw.js', 'r', encoding='utf-8') as f:
    text = f.read()

pos_det = text.find('Throughput')
print(text[pos_det-100:pos_det+2000])
