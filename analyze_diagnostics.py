import re

files = [
    'data/diagnostics/attention-is-all-you-need.pymupdf.txt',
    'data/diagnostics/attention-is-all-you-need.pymupdf.pypdf.txt'
]

headings = ['3.2.1', 'Scaled Dot-Product Attention', '3.2.2', 'Multi-Head Attention', '3.5', 'Positional Encoding', '5.3', 'Optimizer']

occurrences = ['scaled dot-product attention', 'sqrt', 'd_k', 'multi-head attention', 'positional encoding', 'recurrence', 'convolution', 'Adam', 'beta_1', 'beta_2', 'warmup_steps', 'inverse square root']

def get_nearest_page_marker(text, index):
    matches = list(re.finditer(r'--- Page (\d+) ---', text))
    nearest = '--- Page 1 ---'
    for m in matches:
        if m.start() <= index:
            nearest = m.group(0)
        else:
            break
    return nearest

for filepath in files:
    print('='*80)
    print(f'ANALYZING FILE: {filepath}')
    print('='*80)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
        
    print('\n[HEADINGS / TERMS WINDOWS]')
    for heading in headings:
        matches = [m for m in re.finditer(re.escape(heading), text, re.IGNORECASE)]
        if not matches:
            print(f'\nHeading: {heading} -> NOT FOUND')
            continue
        
        m = matches[0]
        start_idx = m.start()
        end_idx = m.end()
        
        midpoint = (start_idx + end_idx) // 2
        w_start = max(0, midpoint - 250)
        w_end = min(len(text), midpoint + 250)
        
        if w_start == 0:
            w_end = min(len(text), 500)
        elif w_end == len(text):
            w_start = max(0, len(text) - 500)
            
        excerpt = text[w_start:w_end]
        page_marker = get_nearest_page_marker(text, start_idx)
        print(f'\n--- HEADING: {heading} (First match index: {start_idx}, Nearest Page: {page_marker}) ---')
        print(excerpt)
        print('-'*40)
        
    print('\n[OCCURRENCE CONTEXTS]')
    for term in occurrences:
        matches = [m for m in re.finditer(re.escape(term), text, re.IGNORECASE)]
        print(f'\n--- OCCURRENCES FOR TERM: "{term}" (Count: {len(matches)}) ---')
        for i, m in enumerate(matches):
            start_idx = m.start()
            end_idx = m.end()
            midpoint = (start_idx + end_idx) // 2
            w_start = max(0, start_idx - 125)
            w_end = min(len(text), end_idx + 125)
            
            # Clamp or adjust if needed to be up to 250 characters
            # Width is (end_idx + 125) - (start_idx - 125) = (end_idx - start_idx) + 250, wait, we want "up to 250 chars center-aligned or containing the occurrence with 125 chars padding or just a total length of 250 around occurrence index".
            # Let's make total length 250 centered at midpoint:
            w_start = max(0, midpoint - 125)
            w_end = min(len(text), midpoint + 125)
            if w_start == 0:
                w_end = min(len(text), 250)
            elif w_end == len(text):
                w_start = max(0, len(text) - 250)
                
            excerpt = text[w_start:w_end]
            page_marker = get_nearest_page_marker(text, start_idx)
            # Remove linebreaks for cleaner display of occurrence context (single line context up to 250 characters)
            clean_excerpt = ' '.join(excerpt.split())
            print(f'Occ {i+1} (Page: {page_marker}): {clean_excerpt}')
