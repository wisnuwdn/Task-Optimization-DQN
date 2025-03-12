import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import pandas as pd

def calculate_weighted_euclidean_distance(employee_skills, task_skills, alpha=0.5):
    """
    Calculate the normalized Weighted Euclidean Distance (WED) between an employee and a task.
    """
    employee_skills = np.array(employee_skills)
    task_skills = np.array(task_skills)
    weights = 1 / (1 + alpha * np.maximum(0, employee_skills - task_skills))
    wed = np.sqrt(np.sum(weights * (employee_skills - task_skills) ** 2))
    max_task_skills = np.full(len(task_skills), 5)
    min_employee_skills = np.zeros(len(employee_skills))
    max_weights = 1 / (1 + alpha * np.maximum(0, min_employee_skills - max_task_skills))
    max_wed = np.sqrt(np.sum(max_weights * (min_employee_skills - max_task_skills) ** 2))
    normalized_wed = 1 - (wed / max_wed)
    return normalized_wed

class TaskAssignmentEnv:
    def __init__(self, tasks, employees, max_workload=20, alpha=0.5):
        self.tasks = tasks
        self.employees = employees
        self.max_workload = max_workload
        self.alpha = alpha
        self.num_tasks = len(tasks)
        self.num_employees = len(employees)
        self.state = np.zeros((self.num_tasks, self.num_employees))
        self.workloads = np.zeros(self.num_employees)

    def valid_actions(self, task_idx):
        valid = []
        for i in range(self.num_employees):
            if self.workloads[i] + self.tasks.iloc[task_idx]['story_points'] <= self.max_workload:
                valid.append(i)
        return valid

    def reset(self):
        self.state = np.zeros((self.num_tasks, self.num_employees))
        self.workloads = np.zeros(self.num_employees)
        return self.state

    def step(self, task_idx, employee_idx):
        task = self.tasks.iloc[task_idx]
        employee = self.employees.iloc[employee_idx]
        task_skills = task[3:].values
        employee_skills = employee[2:].values
        similarity_score = calculate_weighted_euclidean_distance(employee_skills, task_skills, self.alpha)
        self.state[task_idx, employee_idx] = 1
        self.workloads[employee_idx] += task['story_points']
        std_workload = np.std(self.workloads)
        workload_balance_score = 1 / (1 + std_workload)
        reward = 0.7 * similarity_score + 0.3 * workload_balance_score
        done = np.all(self.state.sum(axis=1)) or np.any(self.workloads > self.max_workload)
        return self.state, reward, done

class PPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PPOAgent, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.actor(x), self.critic(x)

    def get_action(self, state, valid_actions):
        logits, _ = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        if not valid_actions:
            return 0, torch.tensor([0.0])
        valid_actions = [a for a in valid_actions if a < len(probs)]
        valid_probs = probs[valid_actions]
        action_dist = Categorical(valid_probs)
        sampled_action = action_dist.sample()
        return valid_actions[sampled_action.item()], action_dist.log_prob(sampled_action)

def load_state_with_dimension_check(model, checkpoint_path):
    """
    Load weights from a checkpoint into the model, only for layers with matching dimensions.
    """
    checkpoint = torch.load(checkpoint_path)
    model_state = model.state_dict()
    for name, param in checkpoint.items():
        if name in model_state and model_state[name].size() == param.size():
            model_state[name].copy_(param)
        else:
            print(f"Skipping layer '{name}' due to size mismatch: {param.size()} vs {model_state[name].size()}")
    model.load_state_dict(model_state, strict=False)
    print(f"Loaded weights with dimension check from {checkpoint_path}")

def evaluate_model(agent, tasks_df, employees_df):
    """
    Evaluate the PPO agent on a given dataset, returning an assignments dataframe.
    """
    env = TaskAssignmentEnv(tasks_df, employees_df)
    state = env.reset()
    assignments = []
    task_idx = 0
    while task_idx < env.num_tasks:
        state_tensor = torch.tensor(state.flatten(), dtype=torch.float32)
        valid_actions = env.valid_actions(task_idx)
        with torch.no_grad():
            action, _ = agent.get_action(state_tensor, valid_actions)
        next_state, _, done = env.step(task_idx, action)
        task = env.tasks.iloc[task_idx]
        employee = env.employees.iloc[action]
        similarity_score = calculate_weighted_euclidean_distance(employee[2:].values, task[3:].values)
        assignments.append({
            'task_id': task['task_id'],
            'project_id': task['project_id'],
            'employee_id': employee['employee_id'],
            'story_points': task['story_points'],
            'similarity_score': similarity_score
        })
        state = next_state
        task_idx += 1
        if done:
            break
    return pd.DataFrame(assignments)