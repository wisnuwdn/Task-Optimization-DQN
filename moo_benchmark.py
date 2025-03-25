import pandas as pd
import numpy as np
from gurobipy import *
import matplotlib.pyplot as plt

def calculate_weighted_euclidean_distance(emp_skills, task_skills, a=0.5):
    """Calculate weighted Euclidean distance with global normalization"""
    diff = emp_skills - task_skills
    w = 1 / (1 + a * np.maximum(0, diff))
    squared_diff = w * (diff ** 2)
    return np.sqrt(np.sum(squared_diff[task_skills > 0]))

class MOOBenchmark:
    def __init__(self, max_workload=20):
        # Load and preprocess data
        self.max_workload = max_workload
        self.load_data()
        self.calculate_similarity_scores()
        
    def load_data(self):
        # Load from Colab working directory
        emp_df = pd.read_csv('subset_employee.csv').fillna(0)
        task_df = pd.read_csv('subset_task.csv').fillna(0)
        
        # Process employee data
        self.employee_skills = emp_df.drop(columns=['employee_id', 'Role']).values
        self.num_employees = len(emp_df)
        
        # Process task data
        self.story_points = task_df['story_points'].values
        self.task_skills = task_df.drop(columns=['task_name', 'epic', 'story_points']).values
        self.num_tasks = len(task_df)
        
        print(f"Loaded {self.num_employees} employees and {self.num_tasks} tasks")
    
    def calculate_similarity_scores(self):
        self.similarity_scores = np.zeros((self.num_tasks, self.num_employees))
        for i in range(self.num_tasks):
            for j in range(self.num_employees):
                self.similarity_scores[i, j] = calculate_weighted_euclidean_distance(
                    self.employee_skills[j], self.task_skills[i]
                )
    
    def optimize(self):
        # Create model
        model = Model("task_assignment")
        
        # Decision variables
        x = {}
        for i in range(self.num_tasks):
            for j in range(self.num_employees):
                x[i,j] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')
        
        # Workload variable
        max_workload_var = model.addVar(vtype=GRB.CONTINUOUS, name='max_workload')
        
        # Constraints
        # Each task assigned to one employee
        for i in range(self.num_tasks):
            model.addConstr(quicksum(x[i,j] for j in range(self.num_employees)) == 1)
        
        # Workload constraints
        for j in range(self.num_employees):
            model.addConstr(
                quicksum(self.story_points[i] * x[i,j] for i in range(self.num_tasks))
                <= self.max_workload
            )
            model.addConstr(
                quicksum(self.story_points[i] * x[i,j] for i in range(self.num_tasks))
                <= max_workload_var
            )
        
        # Objective components
        idle_obj = quicksum(
            1 - quicksum(x[i,j] for i in range(self.num_tasks)) 
            for j in range(self.num_employees)
        )
        
        skill_obj = quicksum(
            self.similarity_scores[i,j] * x[i,j]
            for i in range(self.num_tasks)
            for j in range(self.num_employees)
        )
        
        # Multi-objective with weights
        model.setObjective(
            0.05 * idle_obj + 
            0.85 * skill_obj + 
            0.10 * max_workload_var,
            GRB.MINIMIZE
        )
        
        # Optimize
        model.optimize()
        
        # Extract results
        assignments = np.zeros((self.num_tasks, self.num_employees))
        if model.status == GRB.OPTIMAL:
            for i in range(self.num_tasks):
                for j in range(self.num_employees):
                    if x[i,j].X > 0.5:
                        assignments[i,j] = 1
        
        return assignments

    def evaluate_solution(self, assignments):
        # Calculate metrics
        workloads = np.sum(assignments * self.story_points[:, np.newaxis], axis=0)
        active_employees = np.sum(np.any(assignments, axis=0))
        skill_scores = np.sum(assignments * self.similarity_scores)
        workload_var = np.var(workloads[workloads > 0])
        
        return {
            'active_employees': active_employees,
            'avg_skill_score': skill_scores / self.num_tasks,
            'workload_variance': workload_var,
            'assignments': assignments,
            'workloads': workloads
        }

if __name__ == "__main__":
    # First create the subset
    from create_subset import create_subset_data
    create_subset_data()
    
    # Then run the benchmark
    benchmark = MOOBenchmark()
    assignments = benchmark.optimize()
    results = benchmark.evaluate_solution(assignments)
    
    # Create output directory in Colab
    !mkdir -p output
    
    # Print and visualize results
    print("\nResults:")
    print(f"Active Employees: {results['active_employees']}/{benchmark.num_employees}")
    print(f"Average Skill Score: {results['avg_skill_score']:.4f}")
    print(f"Workload Variance: {results['workload_variance']:.4f}")