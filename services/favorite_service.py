from models.favorite_model import get_favorite_by_user_db, toggle_favorite_db

def get_favorite_by_user(user_id: int):
    return get_favorite_by_user_db(user_id)

def toggle_favorite(user_id: int, product_id: str):
    return toggle_favorite_db(user_id, product_id)