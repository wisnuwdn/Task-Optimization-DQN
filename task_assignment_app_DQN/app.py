# File ini (app.py) adalah file utama aplikasi Task Assignment System, sebuah aplikasi berbasis web menggunakan Streamlit untuk mengelola penugasan tugas kepada karyawan dengan bantuan AI (Deep Q-Learning). Aplikasi ini memungkinkan pengguna untuk login, mengunggah data tugas dan karyawan, menjalankan penugasan otomatis, melihat daftar tugas di Project Board, dan menganalisis hasil penugasan melalui Dashboard.
import streamlit as st  # Impor library Streamlit untuk membuat aplikasi web interaktif.
import pandas as pd  # Impor library Pandas untuk mengelola data dalam bentuk DataFrame (tabel).
import plotly.express as px  # Impor Plotly Express untuk membuat visualisasi data seperti grafik batang.
import plotly.graph_objects as go  # Impor Plotly Graph Objects untuk visualisasi data yang lebih kustom, seperti box plot.
from sqlalchemy.orm import Session  # Impor kelas Session dari SQLAlchemy untuk mengelola sesi database.
from models import User, Task, Employee, TaskSkill, EmployeeSkill, Assignment  # Impor model database (User, Task, dll.) dari file models.py untuk berinteraksi dengan tabel database.
from config import get_db, engine  # Impor fungsi get_db dan engine dari config.py untuk konfigurasi database.
from rl_model import DQNAgent, load_state_with_dimension_check, evaluate_model  # Impor fungsi dan kelas dari rl_model.py untuk menjalankan model AI berbasis Deep Q-Learning.
import logging  # Impor library logging untuk mencatat aktivitas aplikasi (log) ke terminal.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  # Konfigurasi logging dengan level INFO dan format yang mencakup waktu, level, dan pesan.
logger = logging.getLogger(__name__)  # Membuat logger dengan nama modul saat ini untuk digunakan di seluruh kode.

st.set_page_config(page_title="Task Assignment System", layout="wide")  # Mengatur konfigurasi halaman Streamlit: judul halaman "Task Assignment System" dan tata letak "wide" agar lebih lebar.

# Inisialisasi variabel sesi (session state) untuk menyimpan status aplikasi.
if 'logged_in' not in st.session_state:  # Memeriksa apakah 'logged_in' sudah ada di session state.
    st.session_state.logged_in = False  # Jika belum ada, inisialisasi 'logged_in' sebagai False (pengguna belum login).
if 'user_id' not in st.session_state:  # Memeriksa apakah 'user_id' sudah ada di session state.
    st.session_state.user_id = None  # Jika belum ada, inisialisasi 'user_id' sebagai None (belum ada ID pengguna).
if 'assignments' not in st.session_state:  # Memeriksa apakah 'assignments' sudah ada di session state.
    st.session_state.assignments = None  # Jika belum ada, inisialisasi 'assignments' sebagai None (belum ada data penugasan).
if 'tasks_df' not in st.session_state:  # Memeriksa apakah 'tasks_df' sudah ada di session state.
    st.session_state.tasks_df = None  # Jika belum ada, inisialisasi 'tasks_df' sebagai None (belum ada data tugas).
if 'employees_df' not in st.session_state:  # Memeriksa apakah 'employees_df' sudah ada di session state.
    st.session_state.employees_df = None  # Jika belum ada, inisialisasi 'employees_df' sebagai None (belum ada data karyawan).
if 'running_assignment' not in st.session_state:  # Memeriksa apakah 'running_assignment' sudah ada di session state.
    st.session_state.running_assignment = False  # Jika belum ada, inisialisasi 'running_assignment' sebagai False (proses penugasan belum berjalan).
if 'progress_percentage' not in st.session_state:  # Memeriksa apakah 'progress_percentage' sudah ada di session state.
    st.session_state.progress_percentage = 0.0  # Jika belum ada, inisialisasi 'progress_percentage' sebagai 0.0 (progress penugasan 0%).
if 'progress_status' not in st.session_state:  # Memeriksa apakah 'progress_status' sudah ada di session state.
    st.session_state.progress_status = ""  # Jika belum ada, inisialisasi 'progress_status' sebagai string kosong (status progress belum ada).

def login_page():  # Fungsi untuk menampilkan halaman login.
    st.title("Task Assignment System - Login")  # Menampilkan judul halaman login.
    with st.form("login_form"):  # Membuat form login dengan ID "login_form".
        email = st.text_input("Email")  # Menampilkan input teks untuk email pengguna.
        password = st.text_input("Password", type="password")  # Menampilkan input teks untuk password (disembunyikan).
        submit = st.form_submit_button("Login")  # Menampilkan tombol "Login" untuk mengirimkan form.
        
        if submit:  # Jika tombol "Login" ditekan.
            db = Session(engine)  # Membuka sesi database menggunakan engine dari config.py.
            user = db.query(User).filter(User.email == email, User.password == password).first()  # Mencari pengguna di tabel User dengan email dan password yang sesuai.
            db.close()  # Menutup sesi database setelah query selesai.
            
            if user:  # Jika pengguna ditemukan (login berhasil).
                st.session_state.logged_in = True  # Mengatur status login menjadi True di session state.
                st.session_state.user_id = user.id  # Menyimpan ID pengguna ke session state.
                logger.info(f"User logged in with ID: {st.session_state.user_id}")  # Mencatat ke log bahwa pengguna berhasil login dengan ID tertentu.
                
                assignments = db.query(Assignment).join(Task).filter(Task.user_id == user.id).all()  # Mengambil data penugasan dari database untuk pengguna ini.
                if assignments:  # Jika ada data penugasan.
                    assignments_data = []  # Membuat list kosong untuk menyimpan data penugasan.
                    for assignment in assignments:  # Iterasi melalui setiap penugasan.
                        task = assignment.task  # Mengambil data tugas terkait penugasan.
                        assignments_data.append({  # Menambahkan data penugasan ke list dalam bentuk dictionary.
                            'task_id': task.task_id,  # ID tugas.
                            'project_id': task.project_id,  # ID proyek.
                            'story_points': task.story_points,  # Story points tugas.
                            'employee_id': assignment.employee.employee_id,  # ID karyawan yang ditugaskan.
                            'similarity_score': assignment.similarity_score  # Skor kesesuaian penugasan.
                        })
                    st.session_state.assignments = pd.DataFrame(assignments_data)  # Menyimpan data penugasan ke session state sebagai DataFrame.
                
                st.rerun()  # Memuat ulang aplikasi untuk memperbarui tampilan setelah login berhasil.
            else:  # Jika pengguna tidak ditemukan (login gagal).
                st.error("Invalid email or password")  # Menampilkan pesan error bahwa email atau password salah.

def process_uploaded_files(tasks_file, employees_file):  # Fungsi untuk memproses file CSV tugas dan karyawan yang diunggah.
    try:  # Memulai blok try untuk menangani error saat memproses file.
        tasks_df = pd.read_csv(tasks_file)  # Membaca file CSV tugas ke dalam DataFrame.
        employees_df = pd.read_csv(employees_file)  # Membaca file CSV karyawan ke dalam DataFrame.
        
        logger.info(f"Uploaded {len(tasks_df)} tasks and {len(employees_df)} employees")  # Mencatat jumlah tugas dan karyawan yang diunggah ke log.
        
        task_skill_cols = tasks_df.columns[3:].tolist()  # Mengambil kolom keahlian tugas (kolom ke-4 dan seterusnya).
        employee_start_idx = 3 if 'No' in employees_df.columns else 2  # Menentukan indeks awal kolom keahlian karyawan (kolom ke-3 jika ada kolom 'No', jika tidak kolom ke-2).
        employee_skill_cols = employees_df.columns[employee_start_idx:].tolist()  # Mengambil kolom keahlian karyawan.
        common_skills = set(task_skill_cols).intersection(set(employee_skill_cols))  # Mencari keahlian yang sama antara tugas dan karyawan.
        
        if not common_skills:  # Jika tidak ada keahlian yang sama antara tugas dan karyawan.
            st.error("Uploaded files are incompatible: no matching skills between tasks and employees")  # Menampilkan pesan error bahwa file tidak kompatibel.
            return None, None  # Mengembalikan None untuk tasks_df dan employees_df.
        
        if len(task_skill_cols) != len(employee_skill_cols):  # Jika jumlah kolom keahlian tugas dan karyawan tidak sama.
            st.error(f"Skill dimension mismatch: Tasks have {len(task_skill_cols)} skills, Employees have {len(employee_skill_cols)} skills")  # Menampilkan pesan error tentang ketidaksesuaian dimensi keahlian.
            return None, None  # Mengembalikan None untuk tasks_df dan employees_df.

        db = Session(engine)  # Membuka sesi database.
        
        try:  # Memulai blok try untuk menangani error saat menyimpan data ke database.
            user_tasks = db.query(Task).filter(Task.user_id == st.session_state.user_id).all()  # Mengambil semua tugas pengguna saat ini dari database.
            for task in user_tasks:  # Iterasi melalui setiap tugas pengguna.
                db.query(TaskSkill).filter(TaskSkill.task_id == task.id).delete()  # Menghapus data keahlian tugas terkait.
                db.query(Assignment).filter(Assignment.task_id == task.id).delete()  # Menghapus data penugasan terkait tugas.
            db.query(Task).filter(Task.user_id == st.session_state.user_id).delete()  # Menghapus semua tugas pengguna dari database.

            user_employees = db.query(Employee).filter(Employee.user_id == st.session_state.user_id).all()  # Mengambil semua karyawan pengguna saat ini dari database.
            for employee in user_employees:  # Iterasi melalui setiap karyawan pengguna.
                db.query(EmployeeSkill).filter(EmployeeSkill.employee_id == employee.id).delete()  # Menghapus data keahlian karyawan terkait.
                db.query(Assignment).filter(Assignment.employee_id == employee.id).delete()  # Menghapus data penugasan terkait karyawan.
            db.query(Employee).filter(Employee.user_id == st.session_state.user_id).delete()  # Menghapus semua karyawan pengguna dari database.

            logger.info(f"Deleted existing tasks and employees for user {st.session_state.user_id}")  # Mencatat bahwa data lama telah dihapus dari database.

            for _, row in tasks_df.iterrows():  # Iterasi melalui setiap baris di DataFrame tugas.
                task = Task(  # Membuat objek Task baru dengan data dari baris saat ini.
                    task_id=row['task_id'],  # ID tugas.
                    project_id=row['project_id'],  # ID proyek.
                    story_points=row['story_points'],  # Story points tugas.
                    user_id=st.session_state.user_id  # ID pengguna yang mengunggah.
                )
                db.add(task)  # Menambahkan tugas ke sesi database.
                db.flush()  # Menyimpan tugas ke database agar ID tugas tersedia untuk digunakan.
                
                for skill in common_skills:  # Iterasi melalui keahlian yang sama antara tugas dan karyawan.
                    if row[skill] > 0:  # Jika level keahlian untuk tugas ini lebih dari 0.
                        task_skill = TaskSkill(  # Membuat objek TaskSkill baru.
                            task_id=task.id,  # ID tugas yang baru disimpan.
                            skill_name=skill,  # Nama keahlian.
                            skill_level=row[skill]  # Level keahlian dari file CSV.
                        )
                        db.add(task_skill)  # Menambahkan keahlian tugas ke sesi database.
            
            for _, row in employees_df.iterrows():  # Iterasi melalui setiap baris di DataFrame karyawan.
                employee = Employee(  # Membuat objek Employee baru dengan data dari baris saat ini.
                    employee_id=row['employee_id'],  # ID karyawan.
                    role=row['Role'],  # Peran karyawan.
                    user_id=st.session_state.user_id  # ID pengguna yang mengunggah.
                )
                db.add(employee)  # Menambahkan karyawan ke sesi database.
                db.flush()  # Menyimpan karyawan ke database agar ID karyawan tersedia untuk digunakan.
                
                for skill in common_skills:  # Iterasi melalui keahlian yang sama antara tugas dan karyawan.
                    if row[skill] > 0:  # Jika level keahlian untuk karyawan ini lebih dari 0.
                        employee_skill = EmployeeSkill(  # Membuat objek EmployeeSkill baru.
                            employee_id=employee.id,  # ID karyawan yang baru disimpan.
                            skill_name=skill,  # Nama keahlian.
                            skill_level=row[skill]  # Level keahlian dari file CSV.
                        )
                        db.add(employee_skill)  # Menambahkan keahlian karyawan ke sesi database.
            
            db.commit()  # Menyimpan semua perubahan ke database.
            tasks_count = db.query(Task).filter(Task.user_id == st.session_state.user_id).count()  # Menghitung jumlah tugas yang disimpan untuk pengguna ini.
            employees_count = db.query(Employee).filter(Employee.user_id == st.session_state.user_id).count()  # Menghitung jumlah karyawan yang disimpan untuk pengguna ini.
            logger.info(f"Stored {tasks_count} tasks and {employees_count} employees for user {st.session_state.user_id}")  # Mencatat jumlah tugas dan karyawan yang disimpan ke log.
            st.success("Files uploaded and data stored successfully!")  # Menampilkan pesan sukses bahwa file berhasil diunggah.
            
            st.session_state.tasks_df = tasks_df  # Menyimpan DataFrame tugas ke session state.
            st.session_state.employees_df = employees_df  # Menyimpan DataFrame karyawan ke session state.
            st.session_state.assignments = None  # Mengatur ulang data penugasan di session state menjadi None.
            
            return tasks_df, employees_df  # Mengembalikan DataFrame tugas dan karyawan.
        
        except Exception as e:  # Menangkap error saat menyimpan ke database.
            db.rollback()  # Membatalkan semua perubahan jika terjadi error.
            logger.error(f"Error saving to database: {str(e)}")  # Mencatat error ke log.
            raise e  # Melempar ulang error untuk debugging.
        finally:  # Blok yang selalu dijalankan, baik ada error atau tidak.
            db.close()  # Menutup sesi database.
            
    except Exception as e:  # Menangkap error saat memproses file.
        logger.error(f"Error processing files: {str(e)}")  # Mencatat error ke log.
        st.error(f"Error processing files: {str(e)}")  # Menampilkan pesan error kepada pengguna.
        return None, None  # Mengembalikan None untuk tasks_df dan employees_df.

def load_data_from_db():  # Fungsi untuk memuat data tugas dan karyawan dari database.
    db = Session(engine)  # Membuka sesi database.
    try:  # Memulai blok try untuk menangani error saat memuat data.
        tasks = db.query(Task).filter(Task.user_id == st.session_state.user_id).all()  # Mengambil semua tugas pengguna dari database.
        employees = db.query(Employee).filter(Employee.user_id == st.session_state.user_id).all()  # Mengambil semua karyawan pengguna dari database.
        
        logger.info(f"Loaded {len(tasks)} tasks and {len(employees)} employees for user {st.session_state.user_id}")  # Mencatat jumlah tugas dan karyawan yang dimuat ke log.
        
        if not tasks:  # Jika tidak ada tugas yang ditemukan.
            logger.warning("No tasks found in the database for the user.")  # Mencatat peringatan ke log.
            return None, None  # Mengembalikan None untuk tasks_df dan employees_df.
        if not employees:  # Jika tidak ada karyawan yang ditemukan.
            logger.warning("No employees found in the database for the user.")  # Mencatat peringatan ke log.
            return None, None  # Mengembalikan None untuk tasks_df dan employees_df.
        
        tasks_data = []  # Membuat list kosong untuk menyimpan data tugas.
        employees_data = []  # Membuat list kosong untuk menyimpan data karyawan.
        skill_set = set()  # Membuat set kosong untuk menyimpan semua nama keahlian.
        
        for task in tasks:  # Iterasi melalui setiap tugas.
            task_data = {  # Membuat dictionary untuk menyimpan data tugas.
                'task_id': task.task_id,  # ID tugas.
                'project_id': task.project_id,  # ID proyek.
                'story_points': task.story_points  # Story points tugas.
            }
            for skill in task.skills:  # Iterasi melalui keahlian tugas.
                skill_set.add(skill.skill_name)  # Menambahkan nama keahlian ke set.
                task_data[skill.skill_name] = skill.skill_level  # Menambahkan level keahlian ke dictionary tugas.
            tasks_data.append(task_data)  # Menambahkan data tugas ke list.
        
        for employee in employees:  # Iterasi melalui setiap karyawan.
            employee_data = {  # Membuat dictionary untuk menyimpan data karyawan.
                'employee_id': employee.employee_id,  # ID karyawan.
                'Role': employee.role  # Peran karyawan.
            }
            for skill in employee.skills:  # Iterasi melalui keahlian karyawan.
                skill_set.add(skill.skill_name)  # Menambahkan nama keahlian ke set.
                employee_data[skill.skill_name] = skill.skill_level  # Menambahkan level keahlian ke dictionary karyawan.
            employees_data.append(employee_data)  # Menambahkan data karyawan ke list.
        
        tasks_df = pd.DataFrame(tasks_data)  # Mengonversi list data tugas ke DataFrame.
        employees_df = pd.DataFrame(employees_data)  # Mengonversi list data karyawan ke DataFrame.
        
        for skill in skill_set:  # Iterasi melalui semua keahlian yang ditemukan.
            if skill not in tasks_df.columns:  # Jika keahlian tidak ada di kolom DataFrame tugas.
                tasks_df[skill] = 0  # Tambahkan kolom keahlian dengan nilai 0.
            if skill not in employees_df.columns:  # Jika keahlian tidak ada di kolom DataFrame karyawan.
                employees_df[skill] = 0  # Tambahkan kolom keahlian dengan nilai 0.
        
        logger.info("Data loaded successfully into DataFrames")  # Mencatat bahwa data berhasil dimuat ke DataFrame.
        return tasks_df, employees_df  # Mengembalikan DataFrame tugas dan karyawan.
    
    except Exception as e:  # Menangkap error saat memuat data.
        logger.error(f"Error loading data from database: {str(e)}")  # Mencatat error ke log.
        return None, None  # Mengembalikan None untuk tasks_df dan employees_df.
    finally:  # Blok yang selalu dijalankan.
        db.close()  # Menutup sesi database.

def store_assignments(assignments_df):  # Fungsi untuk menyimpan data penugasan ke database.
    db = Session(engine)  # Membuka sesi database.
    try:  # Memulai blok try untuk menangani error saat menyimpan.
        user_tasks = db.query(Task).filter(Task.user_id == st.session_state.user_id).all()  # Mengambil semua tugas pengguna dari database.
        task_ids = [task.id for task in user_tasks]  # Membuat list ID tugas pengguna.
        if task_ids:  # Jika ada tugas.
            db.query(Assignment).filter(Assignment.task_id.in_(task_ids)).delete(synchronize_session=False)  # Menghapus semua penugasan terkait tugas pengguna.
        
        for _, row in assignments_df.iterrows():  # Iterasi melalui setiap baris di DataFrame penugasan.
            task = db.query(Task).filter(Task.task_id == row['task_id'], Task.user_id == st.session_state.user_id).first()  # Mencari tugas berdasarkan task_id dan user_id.
            employee = db.query(Employee).filter(Employee.employee_id == row['employee_id'], Employee.user_id == st.session_state.user_id).first()  # Mencari karyawan berdasarkan employee_id dan user_id.
            
            if task and employee:  # Jika tugas dan karyawan ditemukan.
                assignment = Assignment(  # Membuat objek Assignment baru.
                    task_id=task.id,  # ID tugas.
                    employee_id=employee.id,  # ID karyawan.
                    similarity_score=row['similarity_score'],  # Skor kesesuaian penugasan.
                    status='To Do'  # Status penugasan diatur ke "To Do".
                )
                db.add(assignment)  # Menambahkan penugasan ke sesi database.
        
        db.commit()  # Menyimpan semua perubahan ke database.
    except Exception as e:  # Menangkap error saat menyimpan.
        db.rollback()  # Membatalkan semua perubahan jika terjadi error.
        raise e  # Melempar ulang error untuk debugging.
    finally:  # Blok yang selalu dijalankan.
        db.close()  # Menutup sesi database.

def run_task_assignment():  # Fungsi untuk menjalankan proses penugasan tugas menggunakan AI.
    if st.session_state.tasks_df is None or st.session_state.employees_df is None:  # Memeriksa apakah data tugas dan karyawan sudah ada.
        st.error("Please upload tasks and employees data first.")  # Menampilkan pesan error jika data belum diunggah.
        return  # Menghentikan fungsi.

    if not st.session_state.running_assignment:  # Memeriksa apakah proses penugasan belum berjalan.
        logger.info("Starting task assignment in Streamlit")  # Mencatat bahwa proses penugasan dimulai.
        try:  # Memulai blok try untuk menangani error selama proses penugasan.
            agent = DQNAgent(num_tasks=len(st.session_state.tasks_df), num_employees=len(st.session_state.employees_df))  # Membuat agen AI dengan jumlah tugas dan karyawan.
            
            try:  # Memulai blok try untuk memuat model AI.
                load_state_with_dimension_check(agent, 'clean_dqn_model.pth', 'clean_dqn_agent_params.pth')  # Memuat model AI dan parameter dari file.
            except Exception as e:  # Menangkap error saat memuat model.
                logger.error(f"Error loading model: {str(e)}")  # Mencatat error ke log.
                st.error(f"Error loading model: {str(e)}")  # Menampilkan pesan error kepada pengguna.
                st.session_state.progress_status = f"Error loading model: {str(e)}"  # Memperbarui status progress dengan pesan error.
                return  # Menghentikan fungsi.
            
            st.session_state.running_assignment = True  # Mengatur status bahwa proses penugasan sedang berjalan.
            st.session_state.progress_percentage = 0.0  # Mengatur progress menjadi 0%.
            st.session_state.progress_status = "Starting task assignment..."  # Mengatur status progress ke "Starting task assignment...".
            
            with st.sidebar:  # Membuat elemen di sidebar untuk menampilkan progress.
                st.subheader("Task Assignment Progress")  # Menampilkan subjudul "Task Assignment Progress".
                progress_bar = st.empty()  # Membuat placeholder untuk progress bar.
                status_text = st.empty()  # Membuat placeholder untuk teks status.
                progress_bar.progress(0.0)  # Mengatur progress bar ke 0%.
                status_text.text("Starting task assignment...")  # Menampilkan teks status awal.
            
            generator = evaluate_model(agent, st.session_state.tasks_df, st.session_state.employees_df)  # Menjalankan model AI untuk menghasilkan penugasan.
            for progress in generator:  # Iterasi melalui setiap pembaruan progress dari model AI.
                logger.info(f"Received progress: {progress}")  # Mencatat pembaruan progress ke log.
                
                if progress['status'] == 'progress':  # Jika status progress adalah 'progress'.
                    remaining_tasks = progress['remaining_tasks']  # Mengambil jumlah tugas yang tersisa.
                    total_tasks = progress['total_tasks']  # Mengambil total jumlah tugas.
                    step = progress['step']  # Mengambil nomor langkah saat ini.
                    st.session_state.progress_percentage = (total_tasks - remaining_tasks) / total_tasks if total_tasks > 0 else 0.0  # Menghitung persentase progress.
                    st.session_state.progress_status = f"Step {step}: {remaining_tasks} tasks remaining"  # Memperbarui status progress.
                    progress_bar.progress(min(st.session_state.progress_percentage, 1.0))  # Memperbarui progress bar.
                    status_text.text(st.session_state.progress_status)  # Memperbarui teks status.
                    logger.info(f"Progress updated: Step {step}, {remaining_tasks}/{total_tasks} tasks remaining")  # Mencatat pembaruan progress ke log.
                
                elif progress['status'] == 'complete':  # Jika status progress adalah 'complete'.
                    assignments_df = progress['assignments_df']  # Mengambil DataFrame hasil penugasan.
                    message = progress.get('message', '')  # Mengambil pesan tambahan dari model.
                    logger.info(f"Inference complete: {len(assignments_df)} assignments, message: {message}")  # Mencatat bahwa penugasan selesai.
                    if assignments_df is not None and not assignments_df.empty:  # Jika hasil penugasan valid.
                        st.session_state.assignments = assignments_df  # Menyimpan hasil penugasan ke session state.
                        store_assignments(assignments_df)  # Menyimpan penugasan ke database.
                        st.session_state.progress_status = f"Task assignment completed! {message}"  # Memperbarui status progress ke "completed".
                    else:  # Jika hasil penugasan tidak valid.
                        st.session_state.progress_status = f"Error: No valid assignments generated. {message}"  # Memperbarui status progress dengan pesan error.
                    progress_bar.progress(1.0)  # Mengatur progress bar ke 100%.
                    status_text.text(st.session_state.progress_status)  # Memperbarui teks status.
                    st.session_state.running_assignment = False  # Mengatur status bahwa proses penugasan selesai.
                    break  # Menghentikan loop.
                
                elif progress['status'] == 'error' or progress['status'] == 'timeout':  # Jika status progress adalah 'error' atau 'timeout'.
                    message = progress['message']  # Mengambil pesan error.
                    assignments_dict = progress.get('assignments_dict', {})  # Mengambil penugasan parsial jika ada.
                    logger.error(f"Inference error/timeout: {message}")  # Mencatat error ke log.
                    if assignments_dict:  # Jika ada penugasan parsial.
                        partial_assignments = pd.DataFrame(list(assignments_dict.values()))  # Mengonversi penugasan parsial ke DataFrame.
                        st.session_state.assignments = partial_assignments  # Menyimpan penugasan parsial ke session state.
                        store_assignments(partial_assignments)  # Menyimpan penugasan parsial ke database.
                        st.session_state.progress_status = f"{message} Partial assignments ({len(partial_assignments)}) saved."  # Memperbarui status progress.
                    else:  # Jika tidak ada penugasan parsial.
                        st.session_state.progress_status = message  # Memperbarui status progress dengan pesan error.
                    progress_bar.progress(min(st.session_state.progress_percentage, 1.0))  # Memperbarui progress bar.
                    status_text.text(st.session_state.progress_status)  # Memperbarui teks status.
                    st.error(message)  # Menampilkan pesan error kepada pengguna.
                    st.session_state.running_assignment = False  # Mengatur status bahwa proses penugasan selesai.
                    break  # Menghentikan loop.
            
        except Exception as e:  # Menangkap error selama proses penugasan.
            logger.error(f"Error in task assignment: {str(e)}")  # Mencatat error ke log.
            st.session_state.running_assignment = False  # Mengatur status bahwa proses penugasan selesai.
            st.session_state.progress_status = f"Error in task assignment: {str(e)}"  # Memperbarui status progress dengan pesan error.
            with st.sidebar:  # Membuat elemen di sidebar untuk menampilkan status error.
                status_text = st.empty()  # Membuat placeholder untuk teks status.
                status_text.text(st.session_state.progress_status)  # Menampilkan pesan error di sidebar.

def main_page():  # Fungsi untuk menampilkan halaman utama aplikasi setelah login.
    st.title("Task Assignment System")  # Menampilkan judul halaman utama.
    
    with st.sidebar:  # Membuat elemen di sidebar untuk navigasi.
        st.title("Navigation")  # Menampilkan judul "Navigation" di sidebar.
        page = st.radio("Select Page", ["Upload Data", "Project Board", "Dashboard"])  # Menampilkan radio button untuk memilih halaman: Upload Data, Project Board, atau Dashboard.
        
        if st.session_state.running_assignment or st.session_state.progress_status:  # Memeriksa apakah proses penugasan sedang berjalan atau ada status progress.
            st.subheader("Task Assignment Progress")  # Menampilkan subjudul "Task Assignment Progress" di sidebar.
            st.progress(min(st.session_state.progress_percentage, 1.0))  # Menampilkan progress bar dengan nilai persentase.
            st.text(st.session_state.progress_status)  # Menampilkan teks status progress.
        
        if st.button("Run Task Assignment"):  # Menampilkan tombol "Run Task Assignment" di sidebar.
            run_task_assignment()  # Memanggil fungsi run_task_assignment() jika tombol ditekan.
    
    if page == "Upload Data":  # Jika halaman yang dipilih adalah "Upload Data".
        st.header("Upload Data")  # Menampilkan header "Upload Data".
        tasks_file = st.file_uploader("Upload Tasks CSV", type=['csv'])  # Menampilkan uploader untuk file CSV tugas.
        employees_file = st.file_uploader("Upload Employees CSV", type=['csv'])  # Menampilkan uploader untuk file CSV karyawan.
        upload_button = st.button("Upload")  # Menampilkan tombol "Upload" untuk mengunggah file.
        
        if upload_button and tasks_file is not None and employees_file is not None:  # Jika tombol "Upload" ditekan dan kedua file sudah dipilih.
            tasks_df, employees_df = process_uploaded_files(tasks_file, employees_file)  # Memproses file yang diunggah.
            if tasks_df is not None and employees_df is not None:  # Jika proses berhasil (tidak mengembalikan None).
                st.session_state.tasks_df = tasks_df  # Menyimpan DataFrame tugas ke session state.
                st.session_state.employees_df = employees_df  # Menyimpan DataFrame karyawan ke session state.
                st.session_state.assignments = None  # Mengatur ulang data penugasan menjadi None.
    
    elif page == "Project Board":  # Jika halaman yang dipilih adalah "Project Board".
        logger.info(f"Accessing Project Board. Session state: tasks_df={st.session_state.tasks_df is not None}, employees_df={st.session_state.employees_df is not None}, assignments={st.session_state.assignments is not None}")  # Mencatat status session state saat mengakses halaman.
        
        if 'tasks_df' not in st.session_state or 'employees_df' not in st.session_state or st.session_state.tasks_df is None or st.session_state.employees_df is None:  # Memeriksa apakah data tugas dan karyawan ada di session state.
            tasks_df, employees_df = load_data_from_db()  # Memuat data dari database jika belum ada.
            if tasks_df is not None and employees_df is not None:  # Jika data berhasil dimuat.
                st.session_state.tasks_df = tasks_df  # Menyimpan DataFrame tugas ke session state.
                st.session_state.employees_df = employees_df  # Menyimpan DataFrame karyawan ke session state.
            else:  # Jika data gagal dimuat.
                st.error("Error loading tasks or employees data from the database. Please upload data again.")  # Menampilkan pesan error.
                return  # Menghentikan fungsi.
        
        st.header("Project Backlog")  # Menampilkan header "Project Backlog".
        
        if 'backlog_df' not in st.session_state:  # Memeriksa apakah backlog_df sudah ada di session state.
            if st.session_state.assignments is not None and not st.session_state.assignments.empty:  # Memeriksa apakah data penugasan ada dan tidak kosong.
                st.session_state.backlog_df = st.session_state.assignments[['task_id', 'project_id', 'story_points', 'employee_id']].copy()  # Membuat salinan data penugasan untuk backlog.
                st.session_state.backlog_df['Status'] = 'To Do'  # Menambahkan kolom Status dengan nilai default "To Do".
                st.session_state.backlog_df.rename(columns={'employee_id': 'PIC'}, inplace=True)  # Mengganti nama kolom employee_id menjadi PIC.
            else:  # Jika data penugasan belum ada.
                st.info("Please run task assignment from the sidebar first.")  # Menampilkan pesan untuk menjalankan penugasan terlebih dahulu.
                return  # Menghentikan fungsi.
        
        project_names = sorted(st.session_state.backlog_df['project_id'].unique())  # Mengambil daftar nama proyek yang unik dan mengurutkannya.
        employee_names = sorted(st.session_state.employees_df['employee_id'].unique())  # Mengambil daftar nama karyawan yang unik dan mengurutkannya.
        employee_names = ['None'] + list(employee_names)  # Menambahkan opsi "None" ke daftar nama karyawan.
        
        edited_df = st.data_editor(  # Menampilkan tabel interaktif untuk mengedit backlog.
            st.session_state.backlog_df,  # DataFrame yang akan diedit.
            column_config={  # Konfigurasi kolom tabel.
                'task_id': st.column_config.TextColumn('Task Name'),  # Kolom task_id ditampilkan sebagai "Task Name" dengan tipe teks.
                'project_id': st.column_config.SelectboxColumn(  # Kolom project_id ditampilkan sebagai dropdown.
                    'Project Name',  # Nama kolom.
                    options=project_names,  # Opsi dropdown dari daftar nama proyek.
                    required=True  # Wajib diisi.
                ),
                'story_points': st.column_config.SelectboxColumn(  # Kolom story_points ditampilkan sebagai dropdown.
                    'Story Points',  # Nama kolom.
                    options=[1, 2, 3, 5, 8, 13, 20],  # Opsi story points.
                    required=True  # Wajib diisi.
                ),
                'Status': st.column_config.SelectboxColumn(  # Kolom Status ditampilkan sebagai dropdown.
                    'Status',  # Nama kolom.
                    options=['To Do', 'In Progress', 'Done'],  # Opsi status.
                    required=True  # Wajib diisi.
                ),
                'PIC': st.column_config.SelectboxColumn(  # Kolom PIC ditampilkan sebagai dropdown.
                    'PIC',  # Nama kolom.
                    options=employee_names,  # Opsi dropdown dari daftar nama karyawan.
                    required=True  # Wajib diisi.
                )
            },
            hide_index=True,  # Menyembunyikan indeks tabel.
            num_rows="dynamic",  # Memungkinkan pengguna menambah atau menghapus baris.
            use_container_width=True  # Menggunakan lebar penuh kontainer.
        )
        
        st.session_state.backlog_df = edited_df  # Menyimpan tabel yang sudah diedit kembali ke session state.
    
    elif page == "Dashboard":  # Jika halaman yang dipilih adalah "Dashboard".
        logger.info(f"Accessing Dashboard. Session state: tasks_df={st.session_state.tasks_df is not None}, employees_df={st.session_state.employees_df is not None}, assignments={st.session_state.assignments is not None}")  # Mencatat status session state saat mengakses halaman.
        
        if 'tasks_df' not in st.session_state or 'employees_df' not in st.session_state or st.session_state.tasks_df is None or st.session_state.employees_df is None:  # Memeriksa apakah data tugas dan karyawan ada di session state.
            tasks_df, employees_df = load_data_from_db()  # Memuat data dari database jika belum ada.
            if tasks_df is not None and employees_df is not None:  # Jika data berhasil dimuat.
                st.session_state.tasks_df = tasks_df  # Menyimpan DataFrame tugas ke session state.
                st.session_state.employees_df = employees_df  # Menyimpan DataFrame karyawan ke session state.
            else:  # Jika data gagal dimuat.
                st.error("Error loading tasks or employees data from the database. Please upload data again.")  # Menampilkan pesan error.
                return  # Menghentikan fungsi.

        if st.session_state.assignments is not None and not st.session_state.assignments.empty:  # Memeriksa apakah data penugasan ada dan tidak kosong.
            create_visualizations(st.session_state.assignments)  # Memanggil fungsi untuk membuat visualisasi dari data penugasan.
            
            st.download_button(  # Menampilkan tombol untuk mengunduh data penugasan sebagai CSV.
                label="Download Assignments CSV",  # Label tombol.
                data=st.session_state.assignments.to_csv(index=False),  # Data yang akan diunduh (DataFrame penugasan dalam format CSV).
                file_name="task_assignments.csv",  # Nama file yang akan diunduh.
                mime="text/csv"  # Tipe MIME file.
            )
        else:  # Jika data penugasan belum ada.
            st.info("Please run task assignment from the sidebar to generate assignments.")  # Menampilkan pesan untuk menjalankan penugasan terlebih dahulu.

def create_visualizations(assignments_df):  # Fungsi untuk membuat visualisasi data penugasan di Dashboard.
    col1, col2 = st.columns(2)  # Membagi halaman menjadi dua kolom untuk menampilkan grafik.
    
    with col1:  # Kolom pertama.
        st.subheader("Workload Distribution")  # Menampilkan subjudul "Workload Distribution".
        workload_df = assignments_df.groupby('employee_id')['story_points'].sum().reset_index()  # Mengelompokkan data penugasan berdasarkan karyawan dan menjumlahkan story points.
        fig_workload = px.bar(  # Membuat grafik batang untuk distribusi beban kerja.
            workload_df,  # DataFrame yang digunakan.
            x='employee_id',  # Sumbu X adalah ID karyawan.
            y='story_points',  # Sumbu Y adalah total story points.
            title='Total Story Points per Employee'  # Judul grafik.
        )
        st.plotly_chart(fig_workload, use_container_width=True)  # Menampilkan grafik menggunakan lebar penuh kolom.
    
    with col2:  # Kolom kedua.
        st.subheader("Skill Mismatch Scores")  # Menampilkan subjudul "Skill Mismatch Scores".
        fig_skills = go.Figure()  # Membuat objek Figure untuk grafik.
        fig_skills.add_trace(go.Box(  # Menambahkan grafik box plot untuk skor kesesuaian.
            y=assignments_df['similarity_score'],  # Data skor kesesuaian.
            name='Similarity Scores'  # Nama grafik.
        ))
        fig_skills.update_layout(title='Distribution of Similarity Scores')  # Mengatur judul grafik.
        st.plotly_chart(fig_skills, use_container_width=True)  # Menampilkan grafik menggunakan lebar penuh kolom.
    
    st.subheader("Employee Utilization")  # Menampilkan subjudul "Employee Utilization".
    utilization_df = assignments_df['employee_id'].value_counts().reset_index()  # Menghitung jumlah tugas per karyawan.
    utilization_df.columns = ['employee_id', 'task_count']  # Mengganti nama kolom menjadi 'employee_id' dan 'task_count'.
    fig_util = px.bar(  # Membuat grafik batang untuk utilisasi karyawan.
        utilization_df,  # DataFrame yang digunakan.
        x='employee_id',  # Sumbu X adalah ID karyawan.
        y='task_count',  # Sumbu Y adalah jumlah tugas.
        title='Number of Tasks per Employee'  # Judul grafik.
    )
    st.plotly_chart(fig_util, use_container_width=True)  # Menampilkan grafik menggunakan lebar penuh halaman.

def main():  # Fungsi utama untuk menjalankan aplikasi.
    if not st.session_state.logged_in:  # Memeriksa apakah pengguna belum login.
        login_page()  # Memanggil fungsi login_page() untuk menampilkan halaman login.
    else:  # Jika pengguna sudah login.
        try:  # Memulai blok try untuk menangani error saat menjalankan halaman utama.
            main_page()  # Memanggil fungsi main_page() untuk menampilkan halaman utama.
        except NameError as e:  # Menangkap error NameError (biasanya karena variabel tidak ditemukan).
            if 'tasks_df' not in st.session_state or 'employees_df' not in st.session_state or st.session_state.tasks_df is None or st.session_state.employees_df is None:  # Memeriksa apakah data tugas dan karyawan ada.
                tasks_df, employees_df = load_data_from_db()  # Memuat data dari database jika belum ada.
                if tasks_df is not None and employees_df is not None:  # Jika data berhasil dimuat.
                    st.session_state.tasks_df = tasks_df  # Menyimpan DataFrame tugas ke session state.
                    st.session_state.employees_df = employees_df  # Menyimpan DataFrame karyawan ke session state.
                else:  # Jika data gagal dimuat.
                    st.error("Error loading tasks or employees data from the database. Please upload data again.")  # Menampilkan pesan error.
                    return  # Menghentikan fungsi.
            main_page()  # Memanggil kembali fungsi main_page() setelah data dimuat.

if __name__ == "__main__":  # Memeriksa apakah file ini dijalankan langsung (bukan diimpor sebagai modul).
    main()  # Memanggil fungsi main() untuk memulai aplikasi.