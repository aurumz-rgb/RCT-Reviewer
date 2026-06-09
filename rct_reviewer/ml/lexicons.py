import pickle
import re
from rct_reviewer import get_data_path

class Drugbank:
    def __init__(self):
        path = get_data_path('drugbank/drugbank.pck')
        if path.exists():
            with open(path, 'rb') as f:
                self.data = pickle.load(f)
        else:
            self.data = {}

    def contains_drug(self, text):
        tokens = re.split("([^A-Za-z0-9])", text)
        return 1 if self._find_matches(tokens) else 0

    def _find_matches(self, tokens):
        
        last_buffer = [[]]
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            for blist in last_buffer:
                key = "".join(blist + [token_lower])
                if self.data.get(key):
                    return True
        return False