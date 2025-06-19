# File ini digunakan untuk menginisialisasi database, membuat tabel, dan menambahkan pengguna admin default.
# Mengimpor engine dan SessionLocal dari config.py untuk koneksi ke database
from config import engine, SessionLocal
# Mengimpor Base dan User dari models.py untuk membuat tabel dan menangani data pengguna
from models import Base, User

# Fungsi untuk menginisialisasi database
def init_db():
    # Membuat semua tabel yang didefinisikan di models.py (seperti tabel users, tasks, dll.)
    Base.metadata.create_all(bind=engine)
    
    # Membuat sesi database untuk berinteraksi dengan database
    db = SessionLocal()
    
    # Mencari apakah pengguna default (admin) sudah ada di tabel users
    default_user = db.query(User).filter(User.email == "admin@example.com").first()
    
    # Jika pengguna default belum ada
    if not default_user:
        # Membuat pengguna admin baru dengan email dan kata sandi
        default_user = User(
            email="admin@example.com",
            password="admin123"
        )
        # Menambahkan pengguna ke sesi database
        db.add(default_user)
        # Menyimpan perubahan ke database
        db.commit()
        # Mencetak pesan bahwa pengguna admin berhasil dibuat
        print("Default admin user created!")
    
    # Menutup sesi database agar tidak memakan memori
    db.close()
    # Mencetak pesan bahwa database berhasil diinisialisasi
    print("Database initialized successfully!")

# Mengeksekusi fungsi init_db jika file ini dijalankan langsung
if __name__ == "__main__":
    init_db()