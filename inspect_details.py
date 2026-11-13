with open('frontend/dist/assets/index-CkVcxXKw.js', 'r', encoding='utf-8') as f:
    text = f.read()

# Detection component details
pos_det = text.find('/detection/upload')
print("=== DETECTION COMPONENT ===")
print(text[pos_det-200:pos_det+2000])

# Segmentation component details
pos_seg = text.find('/segmentation/click')
print("\n=== SEGMENTATION COMPONENT ===")
print(text[pos_seg-200:pos_seg+2000])

# History component details
pos_hist = text.find('/history/list')
print("\n=== HISTORY COMPONENT ===")
print(text[pos_hist-200:pos_hist+2500])

# Settings component details
pos_sett = text.find('active_model')
print("\n=== SETTINGS COMPONENT ===")
print(text[pos_sett-200:pos_sett+1500])
