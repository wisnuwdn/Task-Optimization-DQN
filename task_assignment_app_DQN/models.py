# File ini mendefinisikan struktur tabel database untuk menyimpan data pengguna, tugas, karyawan, keahlian, dan penugasan.
# Mengimpor tipe data dan fungsi dari SQLAlchemy untuk membuat kolom dan hubungan tabel
from sqlalchemy import Column, Integer, String, Float, ForeignKey, Table
# Mengimpor fungsi untuk membuat hubungan antar tabel
from sqlalchemy.orm import relationship
# Mengimpor kelas dasar Base dari config.py untuk definisi tabel
from config import Base

# Kelas User untuk tabel pengguna
class User(Base):
    # Nama tabel di database
    __tablename__ = 'users'
    
    # Kolom id sebagai kunci utama (unik untuk setiap pengguna)
    id = Column(Integer, primary_key=True, index=True)
    # Kolom email, harus unik untuk setiap pengguna
    email = Column(String, unique=True, index=True)
    # Kolom kata sandi (disimpan sebagai teks, seharusnya dienkripsi di produksi)
    password = Column(String)

# Kelas Task untuk tabel tugas
class Task(Base):
    # Nama tabel di database
    __tablename__ = 'tasks'
    
    # Kolom id sebagai kunci utama
    id = Column(Integer, primary_key=True, index=True)
    # Kolom task_id, unik untuk setiap tugas
    task_id = Column(String, unique=True, index=True)
    # Kolom project_id untuk menyimpan ID proyek terkait
    project_id = Column(String)
    # Kolom story_points untuk tingkat kesulitan tugas
    story_points = Column(Integer)
    # Kolom user_id sebagai kunci asing, menghubungkan tugas ke pengguna
    user_id = Column(Integer, ForeignKey('users.id'))
    
    # Hubungan: menghubungkan tugas ke pengguna (satu pengguna punya banyak tugas)
    user = relationship('User', back_populates='tasks')
    # Hubungan: menghubungkan tugas ke daftar keahlian yang dibutuhkan
    skills = relationship('TaskSkill', back_populates='task')
    # Hubungan: menghubungkan tugas ke penugasan (satu tugas punya satu penugasan)
    assignment = relationship('Assignment', back_populates='task', uselist=False)

# Kelas Employee untuk tabel karyawan
class Employee(Base):
    # Nama tabel di database
    __tablename__ = 'employees'
    
    # Kolom id sebagai kunci utama
    id = Column(Integer, primary_key=True, index=True)
    # Kolom employee_id, unik untuk setiap karyawan
    employee_id = Column(String, unique=True, index=True)
    # Kolom role untuk peran karyawan (misalnya, "Developer")
    role = Column(String)
    # Kolom user_id sebagai kunci asing, menghubungkan karyawan ke pengguna
    user_id = Column(Integer, ForeignKey('users.id'))
    
    # Hubungan: menghubungkan karyawan ke pengguna (satu pengguna punya banyak karyawan)
    user = relationship('User', back_populates='employees')
    # Hubungan: menghubungkan karyawan ke daftar keahlian yang dimiliki
    skills = relationship('EmployeeSkill', back_populates='employee')
    # Hubungan: menghubungkan karyawan ke daftar penugasan
    assignments = relationship('Assignment', back_populates='employee')

# Kelas TaskSkill untuk tabel keahlian yang dibutuhkan tugas
class TaskSkill(Base):
    # Nama tabel di database
    __tablename__ = 'task_skills'
    
    # Kolom id sebagai kunci utama
    id = Column(Integer, primary_key=True, index=True)
    # Kolom task_id sebagai kunci asing, menghubungkan ke tabel tasks
    task_id = Column(Integer, ForeignKey('tasks.id'))
    # Kolom skill_name untuk nama keahlian (misalnya, "Python")
    skill_name = Column(String)
    # Kolom skill_level untuk tingkat keahlian (misalnya, 3.0)
    skill_level = Column(Float)
    
    # Hubungan: menghubungkan keahlian ke tugas terkait
    task = relationship('Task', back_populates='skills')

# Kelas EmployeeSkill untuk tabel keahlian yang dimiliki karyawan
class EmployeeSkill(Base):
    # Nama tabel di database
    __tablename__ = 'employee_skills'
    
    # Kolom id sebagai kunci utama
    id = Column(Integer, primary_key=True, index=True)
    # Kolom employee_id sebagai kunci asing, menghubungkan ke tabel employees
    employee_id = Column(Integer, ForeignKey('employees.id'))
    # Kolom skill_name untuk nama keahlian
    skill_name = Column(String)
    # Kolom skill_level untuk tingkat keahlian
    skill_level = Column(Float)
    
    # Hubungan: menghubungkan keahlian ke karyawan terkait
    employee = relationship('Employee', back_populates='skills')

# Kelas Assignment untuk tabel penugasan tugas ke karyawan
class Assignment(Base):
    # Nama tabel di database
    __tablename__ = 'assignments'
    
    # Kolom id sebagai kunci utama
    id = Column(Integer, primary_key=True, index=True)
    # Kolom task_id sebagai kunci asing, menghubungkan ke tabel tasks, unik karena satu tugas hanya punya satu penugasan
    task_id = Column(Integer, ForeignKey('tasks.id'), unique=True)
    # Kolom employee_id sebagai kunci asing, menghubungkan ke tabel employees
    employee_id = Column(Integer, ForeignKey('employees.id'))
    # Kolom similarity_score untuk skor kecocokan keahlian
    similarity_score = Column(Float)
    # Kolom status untuk status penugasan (misalnya, "To Do")
    status = Column(String, default='To Do')
    
    # Hubungan: menghubungkan penugasan ke tugas terkait
    task = relationship('Task', back_populates='assignment')
    # Hubungan: menghubungkan penugasan ke karyawan terkait
    employee = relationship('Employee', back_populates='assignments')

# Menambahkan hubungan balik ke kelas User
# Hubungan: satu pengguna punya banyak tugas
User.tasks = relationship('Task', back_populates='user')
# Hubungan: satu pengguna punya banyak karyawan
User.employees = relationship('Employee', back_populates='user')