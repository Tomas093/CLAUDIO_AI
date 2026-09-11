c = open(r'C:\Users\Tomas\.gemini\antigravity\brain\f5dc0534-8f69-4250-8d1a-b5cb725a5aa6\scratch\test_all_models.py', encoding='utf-8').read()  
c = c.replace('bbox_inches=\" "tight\', '')  
open(r'C:\Users\Tomas\.gemini\antigravity\brain\f5dc0534-8f69-4250-8d1a-b5cb725a5aa6\scratch\test_all_models.py', 'w', encoding='utf-8').write(c)  
