import os
import json

BASE_PATH = "vector_store"
COLUMNS_FILE = "columns.json"

def _columns_path(user_id: str) -> str:
    return os.path.join(BASE_PATH, user_id, COLUMNS_FILE)

def save_columns(user_id: str, filename: str, columns_data: dict) -> None:
    """Store column index separately for deterministic native extraction."""
    user_path = os.path.join(BASE_PATH, user_id)
    os.makedirs(user_path, exist_ok=True)
    
    p = _columns_path(user_id)
    store = {}
    if os.path.exists(p):
        with open(p, "r") as f:
            store = json.load(f)
            
    store[filename] = columns_data
    
    with open(p, "w") as f:
        json.dump(store, f, indent=2)

def load_columns(user_id: str, filename: str = None) -> dict:
    """Load column indexes for a user, optionally filtered by filename."""
    p = _columns_path(user_id)
    if not os.path.exists(p):
        return {}
        
    with open(p, "r") as f:
        store = json.load(f)
        
    if filename:
        return store.get(filename, {})
    return store
