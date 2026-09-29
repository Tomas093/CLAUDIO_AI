import os
PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../CLAUDIO_AI/claudio_v2
BASE = os.environ.get('CLAUDIO_BASE', os.path.dirname(PACK))                                          # .../CLAUDIO_AI
DATA = os.path.join(PACK, 'data')
WORK = os.path.join(PACK, 'work')
DS = os.environ.get('CLAUDIO_DS', os.path.join(WORK, 'ds'))
