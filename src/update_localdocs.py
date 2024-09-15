from app import add_local_docs_to_db, create_or_load_db


def update_db(db):
    add_local_docs_to_db(db)

def run():
    db = create_or_load_db()
    update_db(db)

if __name__ == "__main__":
    run()
