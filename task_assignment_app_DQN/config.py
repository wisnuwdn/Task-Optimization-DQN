# File ini berisi pengaturan untuk menghubungkan aplikasi ke database SQLite.
# Mengimpor fungsi dan kelas dari SQLAlchemy untuk mengelola database
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
# Mengimpor modul os untuk menangani path file
import os

# Mendapatkan direktori tempat file ini berada
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Menentukan lokasi database SQLite (file task_assignment.db di direktori yang sama)
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'task_assignment.db')}"

# Membuat engine SQLAlchemy untuk koneksi ke database
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Membuat pabrik sesi untuk mengelola koneksi database
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Membuat kelas dasar untuk model database (digunakan di models.py)
Base = declarative_base()

# Fungsi untuk menyediakan sesi database ke bagian lain aplikasi
def get_db():
    """Mendapatkan sesi database"""
    # Membuat sesi baru
    db = SessionLocal()
    try:
        # Mengembalikan sesi untuk digunakan
        yield db
    finally:
        # Menutup sesi setelah selesai agar tidak memakan memori
        db.close()