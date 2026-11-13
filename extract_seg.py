with open('frontend/dist/assets/index-CkVcxXKw.js', 'r', encoding='utf-8') as f:
    text = f.read()

import re
pos = text.find('/segmentation')
while pos != -1:
    start = max(0, pos - 1500)
    end = min(len(text), pos + 1500)
    print("=== SEGMENTATION SNIPPET ===")
    print(text[start:end])
    pos = text.find('/segmentation', pos + 1)
