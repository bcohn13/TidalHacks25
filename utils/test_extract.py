import sys
import os

# Make sure parent directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.extract_professor import extract_professor_name, load_professor_names

# Now run your test
professors = load_professor_names()
query = "Who's the best for systems programming?"
print("Matched Professor:", extract_professor_name(query, professors))
