import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy.orm import Session
from models import User, Task, Employee, TaskSkill, EmployeeSkill, Assignment
from config import get_db, engine
from rl_model import DQNAgent, load_state_with_dimension_check, evaluate_model
import logging

# Configure logging for Streamlit
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure page settings
st.set_page_config(page_title="Task Assignment System", layout="wide")

# Initialize session state variables
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'assignments' not in st.session_state:
    st.session_state.assignments = None
if 'tasks_df' not in st.session_state:
    st.session_state.tasks_df = None
if 'employees_df' not in st.session_state:
    st.session_state.employees_df = None
if 'running_assignment' not in st.session_state:
    st.session_state.running_assignment = False
if 'progress_percentage' not in st.session_state:
    st.session_state.progress_percentage = 0.0
if 'progress_status' not in st.session_state:
    st.session_state.progress_status = ""

def login_page():
    st.title("Task Assignment System - Login")
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            db = Session(engine)
            user = db.query(User).filter(User.email == email, User.password == password).first()
            db.close()
            
            if user:
                st.session_state.logged_in = True
                st.session_state.user_id = user.id
                
                # Load assignments from database
                assignments = db.query(Assignment).join(Task).filter(Task.user_id == user.id).all()
                if assignments:
                    assignments_data = []
                    for assignment in assignments:
                        task = assignment.task
                        assignments_data.append({
                            'task_id': task.task_id,
                            'project_id': task.project_id,
                            'story_points': task.story_points,
                            'employee_id': assignment.employee.employee_id,
                            'similarity_score': assignment.similarity_score
                        })
                    st.session_state.assignments = pd.DataFrame(assignments_data)
                
                st.rerun()
            else:
                st.error("Invalid email or password")

def process_uploaded_files(tasks_file, employees_file):
    try:
        # Read CSV files
        tasks_df = pd.read_csv(tasks_file)
        employees_df = pd.read_csv(employees_file)
        
        # Extract skill columns based on their position in the CSV
        task_skill_cols = tasks_df.columns[3:].tolist()
        employee_start_idx = 3 if 'No' in employees_df.columns else 2
        employee_skill_cols = employees_df.columns[employee_start_idx:].tolist()
        common_skills = set(task_skill_cols).intersection(set(employee_skill_cols))
        
        if not common_skills:
            st.error("Uploaded files are incompatible: no matching skills between tasks and employees")
            return None, None
        
        # Validate skill dimensions match
        if len(task_skill_cols) != len(employee_skill_cols):
            st.error(f"Skill dimension mismatch: Tasks have {len(task_skill_cols)} skills, Employees have {len(employee_skill_cols)} skills")
            return None, None

        # Store data in database
        db = Session(engine)
        
        try:
            # Clear existing data for the current user
            user_tasks = db.query(Task).filter(Task.user_id == st.session_state.user_id).all()
            for task in user_tasks:
                db.query(TaskSkill).filter(TaskSkill.task_id == task.id).delete()
                db.query(Assignment).filter(Assignment.task_id == task.id).delete()
            db.query(Task).filter(Task.user_id == st.session_state.user_id).delete()

            user_employees = db.query(Employee).filter(Employee.user_id == st.session_state.user_id).all()
            for employee in user_employees:
                db.query(EmployeeSkill).filter(EmployeeSkill.employee_id == employee.id).delete()
                db.query(Assignment).filter(Assignment.employee_id == employee.id).delete()
            db.query(Employee).filter(Employee.user_id == st.session_state.user_id).delete()

            # Store tasks and their skills
            for _, row in tasks_df.iterrows():
                task = Task(
                    task_id=row['task_id'],
                    project_id=row['project_id'],
                    story_points=row['story_points'],
                    user_id=st.session_state.user_id
                )
                db.add(task)
                db.flush()
                
                for skill in common_skills:
                    if row[skill] > 0:
                        task_skill = TaskSkill(
                            task_id=task.id,
                            skill_name=skill,
                            skill_level=row[skill]
                        )
                        db.add(task_skill)
            
            # Store employees and their skills
            for _, row in employees_df.iterrows():
                employee = Employee(
                    employee_id=row['employee_id'],
                    role=row['Role'],
                    user_id=st.session_state.user_id
                )
                db.add(employee)
                db.flush()
                
                for skill in common_skills:
                    if row[skill] > 0:
                        employee_skill = EmployeeSkill(
                            employee_id=employee.id,
                            skill_name=skill,
                            skill_level=row[skill]
                        )
                        db.add(employee_skill)
            
            db.commit()
            st.success("Files uploaded and data stored successfully!")
            
            # Store data in session state but do not run inference
            st.session_state.tasks_df = tasks_df
            st.session_state.employees_df = employees_df
            st.session_state.assignments = None
            
            return tasks_df, employees_df
        
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
            
    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        return None, None

def load_data_from_db():
    db = Session(engine)
    try:
        tasks = db.query(Task).filter(Task.user_id == st.session_state.user_id).all()
        employees = db.query(Employee).filter(Employee.user_id == st.session_state.user_id).all()
        
        if not tasks or not employees:
            return None, None
        
        tasks_data = []
        employees_data = []
        skill_set = set()
        
        for task in tasks:
            task_data = {
                'task_id': task.task_id,
                'project_id': task.project_id,
                'story_points': task.story_points
            }
            for skill in task.skills:
                skill_set.add(skill.skill_name)
                task_data[skill.skill_name] = skill.skill_level
            tasks_data.append(task_data)
        
        for employee in employees:
            employee_data = {
                'employee_id': employee.employee_id,
                'Role': employee.role
            }
            for skill in employee.skills:
                skill_set.add(skill.skill_name)
                employee_data[skill.skill_name] = skill.skill_level
            employees_data.append(employee_data)
        
        tasks_df = pd.DataFrame(tasks_data)
        employees_df = pd.DataFrame(employees_data)
        
        for skill in skill_set:
            if skill not in tasks_df.columns:
                tasks_df[skill] = 0
            if skill not in employees_df.columns:
                employees_df[skill] = 0
        
        return tasks_df, employees_df
    
    finally:
        db.close()

def store_assignments(assignments_df):
    db = Session(engine)
    try:
        # Clear existing assignments for the current user
        user_tasks = db.query(Task).filter(Task.user_id == st.session_state.user_id).all()
        task_ids = [task.id for task in user_tasks]
        if task_ids:
            db.query(Assignment).filter(Assignment.task_id.in_(task_ids)).delete(synchronize_session=False)
        
        for _, row in assignments_df.iterrows():
            task = db.query(Task).filter(Task.task_id == row['task_id'], Task.user_id == st.session_state.user_id).first()
            employee = db.query(Employee).filter(Employee.employee_id == row['employee_id'], Employee.user_id == st.session_state.user_id).first()
            
            if task and employee:
                assignment = Assignment(
                    task_id=task.id,
                    employee_id=employee.id,
                    similarity_score=row['similarity_score'],
                    status='To Do'
                )
                db.add(assignment)
        
        db.commit()
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


def run_task_assignment():
    """
    Run the RL task assignment and update progress in the sidebar.
    """
    if st.session_state.tasks_df is None or st.session_state.employees_df is None:
        st.error("Please upload tasks and employees data first.")
        return

    if not st.session_state.running_assignment:
        logger.info("Starting task assignment in Streamlit")
        try:
            agent = DQNAgent(num_tasks=len(st.session_state.tasks_df), num_employees=len(st.session_state.employees_df))
            
            try:
                load_state_with_dimension_check(agent, 'clean_dqn_model.pth', 'clean_dqn_agent_params.pth')
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                st.error(f"Error loading model: {str(e)}")
                st.session_state.progress_status = f"Error loading model: {str(e)}"
                return
            
            # Initialize progress state
            st.session_state.running_assignment = True
            st.session_state.progress_percentage = 0.0
            st.session_state.progress_status = "Starting task assignment..."
            
            # Create progress bar and status in sidebar
            with st.sidebar:
                st.subheader("Task Assignment Progress")
                progress_bar = st.empty()
                status_text = st.empty()
                progress_bar.progress(0.0)
                status_text.text("Starting task assignment...")
            
            # Run inference and update progress
            generator = evaluate_model(agent, st.session_state.tasks_df, st.session_state.employees_df)
            for progress in generator:
                logger.info(f"Received progress: {progress}")
                
                if progress['status'] == 'progress':
                    remaining_tasks = progress['remaining_tasks']
                    total_tasks = progress['total_tasks']
                    step = progress['step']
                    st.session_state.progress_percentage = (total_tasks - remaining_tasks) / total_tasks if total_tasks > 0 else 0.0
                    st.session_state.progress_status = f"Step {step}: {remaining_tasks} tasks remaining"
                    progress_bar.progress(min(st.session_state.progress_percentage, 1.0))
                    status_text.text(st.session_state.progress_status)
                    logger.info(f"Progress updated: Step {step}, {remaining_tasks}/{total_tasks} tasks remaining")
                
                elif progress['status'] == 'complete':
                    assignments_df = progress['assignments_df']
                    message = progress.get('message', '')
                    logger.info(f"Inference complete: {len(assignments_df)} assignments, message: {message}")
                    if assignments_df is not None and not assignments_df.empty:
                        st.session_state.assignments = assignments_df
                        store_assignments(assignments_df)
                        st.session_state.progress_status = f"Task assignment completed! {message}"
                    else:
                        st.session_state.progress_status = f"Error: No valid assignments generated. {message}"
                    progress_bar.progress(1.0)
                    status_text.text(st.session_state.progress_status)
                    st.session_state.running_assignment = False
                    break
                
                elif progress['status'] == 'error' or progress['status'] == 'timeout':
                    message = progress['message']
                    assignments_dict = progress.get('assignments_dict', {})
                    logger.error(f"Inference error/timeout: {message}")
                    if assignments_dict:
                        partial_assignments = pd.DataFrame(list(assignments_dict.values()))
                        st.session_state.assignments = partial_assignments
                        store_assignments(partial_assignments)
                        st.session_state.progress_status = f"{message} Partial assignments ({len(partial_assignments)}) saved."
                    else:
                        st.session_state.progress_status = message
                    progress_bar.progress(min(st.session_state.progress_percentage, 1.0))
                    status_text.text(st.session_state.progress_status)
                    st.error(message)
                    st.session_state.running_assignment = False
                    break
            
        except Exception as e:
            logger.error(f"Error in task assignment: {str(e)}")
            st.session_state.running_assignment = False
            st.session_state.progress_status = f"Error in task assignment: {str(e)}"
            st.error(f"Error in task assignment: {str(e)}")
            with st.sidebar:
                status_text = st.empty()
                status_text.text(st.session_state.progress_status)


def update_progress():
    """
    Process one step of the inference generator and update progress.
    """
    if st.session_state.inference_generator is None:
        logger.warning("No inference generator found")
        st.session_state.running_assignment = False
        return
    
    logger.info("Updating progress in Streamlit")
    try:
        progress = next(st.session_state.inference_generator)
        logger.info(f"Received progress: {progress}")
        
        if progress['status'] == 'progress':
            remaining_tasks = progress['remaining_tasks']
            total_tasks = progress['total_tasks']
            step = progress['step']
            st.session_state.progress_percentage = (total_tasks - remaining_tasks) / total_tasks if total_tasks > 0 else 0.0
            st.session_state.progress_status = f"Step {step}: {remaining_tasks} tasks remaining"
            logger.info(f"Progress updated: Step {step}, {remaining_tasks}/{total_tasks} tasks remaining")
            st.rerun()  # Force UI update for the next step
            
        elif progress['status'] == 'complete':
            assignments_df = progress['assignments_df']
            message = progress.get('message', '')
            logger.info(f"Inference complete: {len(assignments_df)} assignments, message: {message}")
            if assignments_df is not None and not assignments_df.empty:
                st.session_state.assignments = assignments_df
                store_assignments(assignments_df)
                st.session_state.progress_status = f"Task assignment completed! {message}"
            else:
                st.session_state.progress_status = f"Error: No valid assignments generated. {message}"
            
            # Reset inference state
            st.session_state.running_assignment = False
            st.session_state.inference_generator = None
            st.rerun()
            
        elif progress['status'] == 'error' or progress['status'] == 'timeout':
            message = progress['message']
            assignments_dict = progress.get('assignments_dict', {})
            logger.error(f"Inference error/timeout: {message}")
            if assignments_dict:
                partial_assignments = pd.DataFrame(list(assignments_dict.values()))
                st.session_state.assignments = partial_assignments
                store_assignments(partial_assignments)
                st.session_state.progress_status = f"{message} Partial assignments ({len(partial_assignments)}) saved."
            else:
                st.session_state.progress_status = message
            st.error(message)
            
            # Reset inference state
            st.session_state.running_assignment = False
            st.session_state.inference_generator = None
            st.rerun()
            
    except StopIteration:
        logger.warning("Inference generator stopped unexpectedly")
        st.session_state.running_assignment = False
        st.session_state.inference_generator = None
        st.session_state.progress_status = "Task assignment stopped unexpectedly."
        st.rerun()
    except Exception as e:
        logger.error(f"Error in update_progress: {str(e)}")
        st.session_state.running_assignment = False
        st.session_state.inference_generator = None
        st.session_state.progress_status = f"Error during task assignment: {str(e)}"
        st.error(f"Error during task assignment: {str(e)}")
        st.rerun()

def main_page():
    st.title("Task Assignment System")
    
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Select Page", ["Upload Data", "Project Board", "Dashboard"])
        
        # Display progress bar and status
        if st.session_state.running_assignment or st.session_state.progress_status:
            st.subheader("Task Assignment Progress")
            st.progress(min(st.session_state.progress_percentage, 1.0))
            st.text(st.session_state.progress_status)
        
        # Run Task Assignment button
        if st.button("Run Task Assignment"):
            run_task_assignment()
    
    if page == "Upload Data":
        st.header("Upload Data")
        tasks_file = st.file_uploader("Upload Tasks CSV", type=['csv'])
        employees_file = st.file_uploader("Upload Employees CSV", type=['csv'])
        upload_button = st.button("Upload")
        
        if upload_button and tasks_file is not None and employees_file is not None:
            tasks_df, employees_df = process_uploaded_files(tasks_file, employees_file)
            if tasks_df is not None and employees_df is not None:
                st.session_state.tasks_df = tasks_df
                st.session_state.employees_df = employees_df
                st.session_state.assignments = None
    
    elif page == "Project Board":
        if 'tasks_df' not in st.session_state or 'employees_df' not in st.session_state:
            tasks_df, employees_df = load_data_from_db()
            if tasks_df is not None and employees_df is not None:
                st.session_state.tasks_df = tasks_df
                st.session_state.employees_df = employees_df
            else:
                st.info("Please upload data files in the Upload Data page first.")
                return
        
        st.header("Project Backlog")
        
        if 'backlog_df' not in st.session_state:
            if st.session_state.assignments is not None:
                st.session_state.backlog_df = st.session_state.assignments[['task_id', 'project_id', 'story_points', 'employee_id']].copy()
                st.session_state.backlog_df['Status'] = 'To Do'
                st.session_state.backlog_df.rename(columns={'employee_id': 'PIC'}, inplace=True)
            else:
                st.info("Please run task assignment from the sidebar first.")
                return
        
        project_names = sorted(st.session_state.backlog_df['project_id'].unique())
        employee_names = sorted(st.session_state.employees_df['employee_id'].unique())
        employee_names = ['None'] + list(employee_names)
        
        edited_df = st.data_editor(
            st.session_state.backlog_df,
            column_config={
                'task_id': st.column_config.TextColumn('Task Name'),
                'project_id': st.column_config.SelectboxColumn(
                    'Project Name',
                    options=project_names,
                    required=True
                ),
                'story_points': st.column_config.SelectboxColumn(
                    'Story Points',
                    options=[1, 2, 3, 5, 8, 13, 20],
                    required=True
                ),
                'Status': st.column_config.SelectboxColumn(
                    'Status',
                    options=['To Do', 'In Progress', 'Done'],
                    required=True
                ),
                'PIC': st.column_config.SelectboxColumn(
                    'PIC',
                    options=employee_names,
                    required=True
                )
            },
            hide_index=True,
            num_rows="dynamic",
            use_container_width=True
        )
        
        st.session_state.backlog_df = edited_df
    
    elif page == "Dashboard":
        if 'tasks_df' not in st.session_state or 'employees_df' not in st.session_state:
            tasks_df, employees_df = load_data_from_db()
            if tasks_df is not None and employees_df is not None:
                st.session_state.tasks_df = tasks_df
                st.session_state.employees_df = employees_df
            else:
                st.info("Please upload data files in the Upload Data page first.")
                return

        if st.session_state.assignments is not None and not st.session_state.assignments.empty:
            create_visualizations(st.session_state.assignments)
            
            st.download_button(
                label="Download Assignments CSV",
                data=st.session_state.assignments.to_csv(index=False),
                file_name="task_assignments.csv",
                mime="text/csv"
            )
        else:
            st.info("Please run task assignment from the sidebar to generate assignments.")


def create_visualizations(assignments_df):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Workload Distribution")       
        workload_df = assignments_df.groupby('employee_id')['story_points'].sum().reset_index()
        fig_workload = px.bar(
            workload_df,
            x='employee_id',
            y='story_points',
            title='Total Story Points per Employee'
        )
        st.plotly_chart(fig_workload, use_container_width=True)
    
    with col2:
        st.subheader("Skill Mismatch Scores")
        fig_skills = go.Figure()
        fig_skills.add_trace(go.Box(
            y=assignments_df['similarity_score'],
            name='Similarity Scores'
        ))
        fig_skills.update_layout(title='Distribution of Similarity Scores')
        st.plotly_chart(fig_skills, use_container_width=True)
    
    st.subheader("Employee Utilization")
    utilization_df = assignments_df['employee_id'].value_counts().reset_index()
    utilization_df.columns = ['employee_id', 'task_count']
    fig_util = px.bar(
        utilization_df,
        x='employee_id',
        y='task_count',
        title='Number of Tasks per Employee'
    )
    st.plotly_chart(fig_util, use_container_width=True)

def main():
    if not st.session_state.logged_in:
        login_page()
    else:
        try:
            main_page()
        except NameError as e:
            if 'tasks_df' not in st.session_state or 'employees_df' not in st.session_state:
                tasks_df, employees_df = load_data_from_db()
                if tasks_df is not None and employees_df is not None:
                    st.session_state.tasks_df = tasks_df
                    st.session_state.employees_df = employees_df
                else:
                    st.info("Please upload data files in the Upload Data page first.")
                    return
            main_page()

if __name__ == "__main__":
    main()