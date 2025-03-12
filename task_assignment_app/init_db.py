from config import engine, SessionLocal
from models import Base, User

def init_db():
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    # Create a database session
    db = SessionLocal()
    
    # Check if default user exists
    default_user = db.query(User).filter(User.email == "admin@example.com").first()
    
    if not default_user:
        # Create default admin user
        default_user = User(
            email="admin@example.com",
            password="admin123"  # In production, this should be hashed
        )
        db.add(default_user)
        db.commit()
        print("Default admin user created!")
    
    db.close()
    print("Database initialized successfully!")

if __name__ == "__main__":
    init_db()