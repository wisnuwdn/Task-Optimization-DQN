from google.colab import drive
import pandas as pd
import numpy as np

def create_subset_data(n_employees=10, n_tasks=50):
    """Create subset of the original dataset"""
    # Mount Google Drive
    drive.mount('/content/drive')
    
    # Source paths
    task_data_path = '/content/drive/MyDrive/Skripsi/Resources/Datasets/tasks.csv'
    employee_data_path = '/content/drive/MyDrive/Skripsi/Resources/Datasets/employees.csv'
    
    # Load original data
    df_employees = pd.read_csv(employee_data_path)
    df_tasks = pd.read_csv(task_data_path)
    
    # Get skill columns (all columns except the non-skill ones)
    employee_skill_cols = df_employees.columns[3:]  # Skip No, employee_id, Role
    task_skill_cols = df_tasks.columns[3:]         # Skip task_id, project_id, story_points
    
    # Verify they match
    assert set(employee_skill_cols) == set(task_skill_cols), "Skill columns don't match"
    
    # Select random subset
    emp_subset = df_employees.sample(n=n_employees, random_state=42)
    task_subset = df_tasks.sample(n=n_tasks, random_state=42)
    
    # Save subsets to Colab working directory
    emp_subset.to_csv('subset_employee.csv', index=False)
    task_subset.to_csv('subset_task.csv', index=False)
    
    print(f"Created subset with {n_employees} employees and {n_tasks} tasks")
    print(f"Number of skills: {len(employee_skill_cols)}")
    
    # Print sample of data to verify
    print("\nEmployee skills shape:", emp_subset.iloc[:, 3:].shape)
    print("Task skills shape:", task_subset.iloc[:, 3:].shape)
    
    return emp_subset, task_subset

if __name__ == "__main__":
    emp_subset, task_subset = create_subset_data()