with open('frontend/dist/assets/index-CkVcxXKw.js', 'r', encoding='utf-8') as f:
    text = f.read()

pos_det = text.find('u(e.data)')
print("=== DETECTION RESULT HANDLING ===")
print(text[pos_det-50:pos_det+2500])

pos_seg = text.find('u(e.data)')
pos_seg = text.find('u(e.data)', pos_seg + 1)
print("\n=== SEGMENTATION RESULT HANDLING ===")
print(text[pos_seg-50:pos_seg+2500])
