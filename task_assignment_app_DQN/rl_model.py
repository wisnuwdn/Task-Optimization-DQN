import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random
from gym import Env, spaces
from collections import deque
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global constants
MAX_TASKS = 500
MAX_EMPLOYEES = 200
MAX_WORKLOAD = 20
ALPHA_WED = 0.5

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

class TaskAssignmentEnv(Env):
    def __init__(self, tasks, employees, max_tasks=MAX_TASKS, max_employees=MAX_EMPLOYEES):
        super(TaskAssignmentEnv, self).__init__()
        self.num_tasks = len(tasks)
        self.num_employees = len(employees)
        self.max_tasks = max_tasks
        self.max_employees = max_employees
        self.max_workload = MAX_WORKLOAD

        # Preprocess data
        skill_columns = [col for col in tasks.columns if col not in ['task_id', 'story_points', 'project_id']]
        project_id_map = {pid: idx + 1 for idx, pid in enumerate(tasks['project_id'].unique())}
        project_ids = tasks['project_id'].map(project_id_map).values
        task_skills_df = tasks[skill_columns].fillna(0).astype(np.float32)
        employee_skills_df = employees[skill_columns].fillna(0).astype(np.float32)
        story_points_df = tasks['story_points'].astype(np.float32)

        self.task_skills = F.pad(
            torch.tensor(task_skills_df.values / 5.0, dtype=torch.float32, device=device),
            (0, 0, 0, self.max_tasks - self.num_tasks),
            'constant', 0
        )
        self.employee_skills = F.pad(
            torch.tensor(employee_skills_df.values / 5.0, dtype=torch.float32, device=device),
            (0, 0, 0, self.max_employees - self.num_employees),
            'constant', 0
        )
        self.story_points = F.pad(
            torch.tensor(story_points_df.values, dtype=torch.float32, device=device),
            (0, self.max_tasks - self.num_tasks),
            'constant', 0
        )
        self.project_ids = F.pad(
            torch.tensor(project_ids, dtype=torch.long, device=device),
            (0, self.max_tasks - self.num_tasks),
            'constant', -1
        )
        self.num_skills = len(skill_columns)
        self.tasks_df = tasks
        self.employees_df = employees

        # Define observation and action spaces
        self.observation_space = spaces.Dict({
            'state_matrix': spaces.Box(low=0, high=1, shape=(self.max_tasks, 5), dtype=np.float32),
            'workload_matrix': spaces.Box(low=0, high=MAX_WORKLOAD, shape=(self.max_employees,), dtype=np.float32),
            'global_features': spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32),
            'task_mask': spaces.Box(low=0, high=1, shape=(self.max_tasks,), dtype=np.float32)
        })
        self.action_space = spaces.MultiDiscrete([2, self.num_tasks, self.num_employees])
        self.reset()

    def reset(self):
        self.workloads = torch.zeros(self.max_employees, device=device, dtype=torch.float32)
        self.assignments = torch.full((self.max_tasks,), -1, dtype=torch.long, device=device)
        self.remaining_tasks = torch.arange(self.num_tasks, device=device, dtype=torch.long)
        self.employee_projects = torch.full((self.max_employees,), -1, dtype=torch.long, device=device)
        self.wed_scores = []
        self.reassign_counts = torch.zeros(self.max_tasks, device=device, dtype=torch.long)
        self.total_reassigns = 0
        self._initialize_state_matrix()
        return self._get_obs()

    def _initialize_state_matrix(self):
        task_skills_expanded = self.task_skills[:self.num_tasks].unsqueeze(1).expand(-1, self.max_employees, -1)
        emp_skills_expanded = self.employee_skills.unsqueeze(0).expand(self.num_tasks, -1, -1)
        diff = emp_skills_expanded - task_skills_expanded
        weights = 1 / (1 + ALPHA_WED * torch.maximum(diff, torch.tensor(0.0, device=device)))
        mask = self.task_skills[:self.num_tasks].unsqueeze(1).expand(-1, self.max_employees, -1) > 0
        weighted_diff = weights * mask * (emp_skills_expanded - task_skills_expanded) ** 2
        wed = torch.sqrt(torch.sum(weighted_diff, dim=2))
        num_skills_per_task = torch.sum(self.task_skills[:self.num_tasks] > 0, dim=1).unsqueeze(1).expand(-1, self.max_employees)
        wed_worst = torch.sqrt(num_skills_per_task * (1.0 ** 2))
        wed_worst = torch.where(num_skills_per_task > 0, wed_worst, torch.tensor(1.0, device=device))
        similarity_matrix = (1 - (wed / wed_worst)).to(dtype=torch.float32)
        self.state_matrix = torch.topk(similarity_matrix, k=5, dim=1).values
        self.state_matrix = F.pad(self.state_matrix, (0, 0, 0, self.max_tasks - self.num_tasks), 'constant', 0)

    def _update_state_matrix(self, task_idx, emp_idx):
        task_skills_expanded = self.task_skills[task_idx].unsqueeze(0).expand(self.max_employees, -1)
        emp_skills_expanded = self.employee_skills.expand(self.max_employees, -1)
        diff = emp_skills_expanded - task_skills_expanded
        weights = 1 / (1 + ALPHA_WED * torch.maximum(diff, torch.tensor(0.0, device=device)))
        mask = task_skills_expanded > 0
        weighted_diff = weights * mask * (emp_skills_expanded - task_skills_expanded) ** 2
        wed = torch.sqrt(torch.sum(weighted_diff, dim=1))
        num_skills = torch.sum(mask.float(), dim=1)
        wed_worst = torch.sqrt(num_skills * (1.0 ** 2)) if num_skills[0] > 0 else 1.0
        similarities = 1 - (wed / wed_worst)
        top_k_similarities = torch.topk(similarities, k=5, dim=0).values
        self.state_matrix[task_idx] = top_k_similarities

    def _get_obs(self):
        workload_matrix = self.workloads
        prop_remaining = len(self.remaining_tasks) / self.num_tasks if self.num_tasks > 0 else 0.0
        num_idle = torch.sum(self.workloads[:self.num_employees] == 0).item()
        prop_idle = num_idle / self.num_employees if self.num_employees > 0 else 0.0
        std_workload = torch.std(self.workloads[:self.num_employees]).item() / self.max_workload
        global_features = torch.tensor([prop_remaining, prop_idle, std_workload], device=device, dtype=torch.float32)
        task_mask = torch.zeros(self.max_tasks, device=device, dtype=torch.float32)
        task_mask[self.remaining_tasks] = 1
        return {
            'state_matrix': self.state_matrix,
            'workload_matrix': workload_matrix,
            'global_features': global_features,
            'task_mask': task_mask
        }

    def step(self, action):
        action_type, task_idx, emp_idx = action
        done = False
        reward = {'similarity': 0.0, 'workload': 0.0, 'idle': 0.0}
        info = {'action_valid': True}

        if emp_idx < 0 or emp_idx >= self.num_employees or task_idx < 0 or task_idx >= self.num_tasks:
            reward['similarity'] = -1.0
            info['action_valid'] = False
            return self._get_obs(), reward, done, info

        current_std = torch.std(self.workloads[:self.num_employees]).item()

        if action_type == 0:
            if task_idx not in self.remaining_tasks:
                reward['similarity'] = -1.0
                info['action_valid'] = False
                return self._get_obs(), reward, done, info

            task_project = self.project_ids[task_idx]
            emp_project = self.employee_projects[emp_idx]
            current_workload = self.workloads[emp_idx]
            if emp_project != -1 and emp_project != task_project:
                reward['similarity'] = -1.0
                info['action_valid'] = False
                return self._get_obs(), reward, done, info
            if current_workload + self.story_points[task_idx] > self.max_workload:
                reward['similarity'] = -1.0
                info['action_valid'] = False
                return self._get_obs(), reward, done, info

            was_idle = (self.workloads[emp_idx] == 0)
            self.assignments[task_idx] = emp_idx
            self.workloads[emp_idx] += self.story_points[task_idx]
            self.remaining_tasks = self.remaining_tasks[self.remaining_tasks != task_idx]
            if self.employee_projects[emp_idx] == -1:
                self.employee_projects[emp_idx] = task_project
            self._update_state_matrix(task_idx, emp_idx)

        elif action_type == 1:
            if task_idx in self.remaining_tasks:
                reward['similarity'] = -1.0
                info['action_valid'] = False
                return self._get_obs(), reward, done, info
            if self.reassign_counts[task_idx] >= 3:
                reward['similarity'] = -1.0
                info['action_valid'] = False
                return self._get_obs(), reward, done, info

            current_emp_idx = self.assignments[task_idx]
            if current_emp_idx == emp_idx:
                reward['similarity'] = -1.0
                info['action_valid'] = False
                return self._get_obs(), reward, done, info
            task_project = self.project_ids[task_idx]
            emp_project = self.employee_projects[emp_idx]
            current_workload = self.workloads[emp_idx]
            if emp_project != -1 and emp_project != task_project:
                reward['similarity'] = -1.0
                info['action_valid'] = False
                return self._get_obs(), reward, done, info
            if current_workload + self.story_points[task_idx] > self.max_workload:
                reward['similarity'] = -1.0
                info['action_valid'] = False
                return self._get_obs(), reward, done, info

            self.workloads[current_emp_idx] -= self.story_points[task_idx]
            emp_tasks = (self.assignments == current_emp_idx) & (self.project_ids == task_project)
            emp_tasks[task_idx] = False
            if not torch.any(emp_tasks):
                self.employee_projects[current_emp_idx] = -1
            was_idle = (self.workloads[emp_idx] == 0)
            self.assignments[task_idx] = emp_idx
            self.workloads[emp_idx] += self.story_points[task_idx]
            if self.employee_projects[emp_idx] == -1:
                self.employee_projects[emp_idx] = task_project
            self.reassign_counts[task_idx] += 1
            self.total_reassigns += 1
            self._update_state_matrix(task_idx, emp_idx)

        else:
            reward['similarity'] = -1.0
            info['action_valid'] = False
            return self._get_obs(), reward, done, info

        new_std = torch.std(self.workloads[:self.num_employees]).item()
        std_change = new_std - current_std
        if std_change < 0:
            reward['workload'] += 0.5
        elif std_change > 0:
            reward['workload'] -= 0.3

        task_skills_t = self.task_skills[task_idx]
        emp_skills_e = self.employee_skills[emp_idx]
        diff = emp_skills_e - task_skills_t
        weights = 1 / (1 + ALPHA_WED * torch.maximum(diff, torch.tensor(0.0, device=device)))
        mask = task_skills_t > 0
        weighted_diff = weights[mask] * (emp_skills_e[mask] - task_skills_t[mask]) ** 2
        wed = torch.sqrt(torch.sum(weighted_diff))
        num_skills = torch.sum(mask.float())
        wed_worst = torch.sqrt(num_skills * (1.0 ** 2)) if num_skills > 0 else 1.0
        similarity = 1 - (wed / wed_worst) if wed_worst > 0 else 0.0
        info['similarity'] = similarity.item()
        self.wed_scores.append(wed.item())

        if action_type == 0:
            reward['similarity'] = similarity.item()
            if was_idle:
                reward['idle'] = 0.1
        else:
            reward['similarity'] = 0.0

        if len(self.remaining_tasks) == 0:
            done = True
            assigned_tasks = torch.where(self.assignments != -1)[0]
            total_similarity = 0.0
            for task_idx in assigned_tasks:
                emp_idx = self.assignments[task_idx].item()
                task_skills_t = self.task_skills[task_idx]
                emp_skills_e = self.employee_skills[emp_idx]
                diff = emp_skills_e - task_skills_t
                weights = 1 / (1 + ALPHA_WED * torch.maximum(diff, torch.tensor(0.0, device=device)))
                mask = task_skills_t > 0
                weighted_diff = weights[mask] * (emp_skills_e[mask] - task_skills_t[mask]) ** 2
                wed = torch.sqrt(torch.sum(weighted_diff))
                num_skills = torch.sum(mask.float())
                wed_worst = torch.sqrt(num_skills * (1.0 ** 2)) if num_skills > 0 else 1.0
                similarity = 1 - (wed / wed_worst) if wed_worst > 0 else 0.0
                total_similarity += similarity
            avg_similarity = (total_similarity / self.num_tasks if self.num_tasks > 0 else 0.0).item()
            reward['similarity'] += 50.0 * (2.0 * avg_similarity - 1.0)
            std_workload = torch.std(self.workloads[:self.num_employees]).item()
            max_std = self.max_workload / 2
            normalized_std = std_workload / max_std if max_std > 0 else 0.0
            reward['workload'] = 100.0 * (1.0 - 2.0 * normalized_std)
            num_idle = torch.sum(self.workloads[:self.num_employees] == 0).item()
            idle_ratio = num_idle / self.num_employees
            reward['idle'] += 50.0 * (1.0 - 2.0 * idle_ratio)
            info['avg_similarity'] = avg_similarity
            info['std_workload'] = std_workload
            info['num_idle'] = num_idle

        return self._get_obs(), reward, done, info

class DQN(nn.Module):
    def __init__(self, input_dim, max_employees, max_tasks, num_employees, num_tasks):
        super(DQN, self).__init__()
        self.max_employees = max_employees
        self.max_tasks = max_tasks
        self.num_employees = num_employees
        self.num_tasks = num_tasks
        self.fc_state = nn.Linear(max_tasks * 5, 128).to(device, dtype=torch.float32)
        self.fc_workload = nn.Linear(max_employees, 64).to(device, dtype=torch.float32)
        self.fc_global = nn.Linear(3, 16).to(device, dtype=torch.float32)
        self.fc_combined = nn.Linear(128 + 64 + 16, 256).to(device, dtype=torch.float32)
        self.fc2 = nn.Linear(256, 128).to(device, dtype=torch.float32)
        self.fc3 = nn.Linear(128, 2 * max_tasks * max_employees).to(device, dtype=torch.float32)

    def forward(self, state_matrix, workload_matrix, global_features):
        batch_size = state_matrix.size(0)
        state_input = state_matrix.view(batch_size, -1)
        state_input = F.relu(self.fc_state(state_input))
        workload_input = workload_matrix.view(batch_size, -1)
        workload_input = F.relu(self.fc_workload(workload_input))
        global_input = global_features.view(batch_size, -1)
        global_input = F.relu(self.fc_global(global_input))
        x = torch.cat([state_input, workload_input, global_input], dim=1)
        x = F.relu(self.fc_combined(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x).view(batch_size, 2, self.max_tasks, self.max_employees)
        q_values[:, :, self.num_tasks:, :] = -float('inf')
        q_values[:, :, :, self.num_employees:] = -float('inf')
        return q_values

class DQNAgent:
    def __init__(self, num_tasks, num_employees):
        self.device = device
        self.num_tasks = num_tasks
        self.num_employees = num_employees
        self.max_tasks = MAX_TASKS
        self.max_employees = MAX_EMPLOYEES
        self.model = DQN(1 + 1 + 1 + 4, self.max_employees, self.max_tasks, self.num_employees, self.num_tasks).to(device)
        self.target_model = DQN(1 + 1 + 1 + 4, self.max_employees, self.max_tasks, self.num_employees, self.num_tasks).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.memory = deque(maxlen=50000)
        self.epsilon = 0.05
        self.temperature = 1.0

    def act(self, obs, env):
        valid_tasks = torch.where(obs['task_mask'][:self.num_tasks] == 1)[0]
        assigned_tasks = torch.where(env.assignments[:self.num_tasks] != -1)[0]

        if random.random() < self.epsilon:
            if len(valid_tasks) > 0 and random.random() < 0.8:
                task_idx = random.choice(valid_tasks.cpu().numpy())
                valid_emps = torch.where(
                    (env.workloads[:self.num_employees] + env.story_points[task_idx] <= env.max_workload) &
                    ((env.employee_projects[:self.num_employees] == -1) | (env.employee_projects[:self.num_employees] == env.project_ids[task_idx]))
                )[0]
                if valid_emps.size(0) > 0:
                    emp_idx = random.choice(valid_emps.cpu().numpy())
                    return [0, task_idx, emp_idx]
            else:
                if len(assigned_tasks) == 0:
                    task_idx = random.choice(valid_tasks.cpu().numpy())
                    valid_emps = torch.where(
                        (env.workloads[:self.num_employees] + env.story_points[task_idx] <= env.max_workload) &
                        ((env.employee_projects[:self.num_employees] == -1) | (env.employee_projects[:self.num_employees] == env.project_ids[task_idx]))
                    )[0]
                    if valid_emps.size(0) > 0:
                        emp_idx = random.choice(valid_emps.cpu().numpy())
                        return [0, task_idx, emp_idx]
                task_idx = random.choice(assigned_tasks.cpu().numpy())
                current_emp_idx = env.assignments[task_idx].item()
                valid_emps = torch.where(
                    (torch.arange(self.num_employees, device=device) != current_emp_idx) &
                    (env.workloads[:self.num_employees] + env.story_points[task_idx] <= env.max_workload) &
                    ((env.employee_projects[:self.num_employees] == -1) | (env.employee_projects[:self.num_employees] == env.project_ids[task_idx]))
                )[0]
                if valid_emps.size(0) > 0:
                    emp_idx = random.choice(valid_emps.cpu().numpy())
                    return [1, task_idx, emp_idx]

        state_matrix = obs['state_matrix'].unsqueeze(0).to(dtype=torch.float32)
        workload_matrix = obs['workload_matrix'].unsqueeze(0).to(dtype=torch.float32)
        global_features = obs['global_features'].unsqueeze(0).to(dtype=torch.float32)

        with torch.no_grad():
            q_values = self.model(state_matrix, workload_matrix, global_features)
            task_mask = torch.zeros(self.max_tasks, device=device, dtype=torch.bool)
            task_mask[valid_tasks] = 1
            emp_mask = torch.ones(self.max_tasks, self.max_employees, device=device, dtype=torch.bool)
            for t in range(self.num_tasks):
                emp_mask[t, :self.num_employees] = (
                    (env.workloads[:self.num_employees] + env.story_points[t] <= env.max_workload) &
                    ((env.employee_projects[:self.num_employees] == -1) | (env.employee_projects[:self.num_employees] == env.project_ids[t]))
                )
                emp_mask[t, self.num_employees:] = False
            assign_mask = torch.zeros_like(q_values, dtype=torch.bool)
            assign_mask[:, 0] = task_mask.unsqueeze(-1).expand(-1, self.max_employees) & emp_mask
            reassign_mask = torch.zeros_like(q_values, dtype=torch.bool)
            for t in assigned_tasks:
                if env.reassign_counts[t] < 3:
                    current_emp_idx = env.assignments[t].item()
                    reassign_emp_mask = emp_mask[t].clone()
                    reassign_emp_mask[current_emp_idx] = False
                    reassign_mask[0, 1, t] = reassign_emp_mask
            combined_mask = assign_mask | reassign_mask
            q_values[~combined_mask] = -float('inf')

            if torch.all(q_values == -float('inf')):
                if len(valid_tasks) > 0:
                    task_idx = random.choice(valid_tasks.cpu().numpy())
                    valid_emps = torch.where(
                        (env.workloads[:self.num_employees] + env.story_points[task_idx] <= env.max_workload) &
                        ((env.employee_projects[:self.num_employees] == -1) | (env.employee_projects[:self.num_employees] == env.project_ids[task_idx]))
                    )[0]
                    if valid_emps.size(0) > 0:
                        emp_idx = random.choice(valid_emps.cpu().numpy())
                        return [0, task_idx, emp_idx]
                else:
                    task_idx = random.choice(assigned_tasks.cpu().numpy())
                    current_emp_idx = env.assignments[task_idx].item()
                    valid_emps = torch.where(
                        (torch.arange(self.num_employees, device=device) != current_emp_idx) &
                        (env.workloads[:self.num_employees] + env.story_points[task_idx] <= env.max_workload) &
                        ((env.employee_projects[:self.num_employees] == -1) | (env.employee_projects[:self.num_employees] == env.project_ids[task_idx]))
                    )[0]
                    if valid_emps.size(0) > 0:
                        emp_idx = random.choice(valid_emps.cpu().numpy())
                        return [1, task_idx, emp_idx]
            else:
                action_idx = torch.argmax(q_values.view(-1)).item()
                action_type = action_idx // (self.max_tasks * self.max_employees)
                task_emp_idx = action_idx % (self.max_tasks * self.max_employees)
                task_idx = task_emp_idx // self.max_employees
                emp_idx = task_emp_idx % self.max_employees
                return [action_type, task_idx, emp_idx]

    def load(self, model_path, agent_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        agent_params = torch.load(agent_path, map_location=self.device, weights_only=False)
        self.epsilon = agent_params.get('epsilon', 0.05)
        self.temperature = agent_params.get('temperature', 1.0)

def load_state_with_dimension_check(agent, model_path, agent_path):
    """
    Load weights and agent parameters from checkpoints.
    """
    agent.load(model_path, agent_path)
    print(f"Loaded model from {model_path} and agent parameters from {agent_path}")

def evaluate_model(agent, tasks_df, employees_df):
    """
    Evaluate the DDQN agent on a given dataset, yielding progress updates and returning an assignments dataframe with unique task assignments.
    """
    logger.info("Starting RL inference evaluation")
    try:
        env = TaskAssignmentEnv(tasks_df, employees_df)
        obs = env.reset()
        assignments_dict = {}  # Track latest assignment for each task
        done = False
        step = 0
        max_steps = len(tasks_df) * 10  # Prevent infinite loops
        total_tasks = len(tasks_df)
        step_start_time = time.time()

        while not done and step < max_steps:
            # Check for step timeout (e.g., 10 seconds)
            if time.time() - step_start_time > 10:
                logger.warning(f"Step {step}: Timeout after 10 seconds")
                yield {
                    'status': 'timeout',
                    'message': f"Step {step} timed out after 10 seconds",
                    'assignments_dict': assignments_dict
                }
                return

            logger.info(f"Step {step}: Calling agent.act")
            try:
                action = agent.act(obs, env)
                logger.info(f"Step {step}: Action taken - Type: {action[0]}, Task: {action[1]}, Employee: {action[2]}")
            except Exception as e:
                logger.error(f"Step {step}: Error in agent.act - {str(e)}")
                yield {
                    'status': 'error',
                    'message': f"Error in agent.act: {str(e)}",
                    'assignments_dict': assignments_dict
                }
                return

            logger.info(f"Step {step}: Calling env.step")
            try:
                next_obs, _, done, info = env.step(action)
                logger.info(f"Step {step}: Action valid: {info['action_valid']}, Remaining tasks: {len(env.remaining_tasks)}, Done: {done}")
            except Exception as e:
                logger.error(f"Step {step}: Error in env.step - {str(e)}")
                yield {
                    'status': 'error',
                    'message': f"Error in env.step: {str(e)}",
                    'assignments_dict': assignments_dict
                }
                return

            if info['action_valid']:
                task_idx = action[1]
                emp_idx = action[2]
                task = env.tasks_df.iloc[task_idx]
                employee = env.employees_df.iloc[emp_idx]
                assignments_dict[task['task_id']] = {
                    'task_id': task['task_id'],
                    'project_id': task['project_id'],
                    'employee_id': employee['employee_id'],
                    'story_points': task['story_points'],
                    'similarity_score': info['similarity']
                }
            
            logger.info(f"Step {step}: Yielding progress")
            yield {
                'status': 'progress',
                'remaining_tasks': len(env.remaining_tasks),
                'total_tasks': total_tasks,
                'step': step
            }
            
            logger.info(f"Step {step}: Updating state for next step")
            try:
                obs = next_obs
                step += 1
                step_start_time = time.time()
            except Exception as e:
                logger.error(f"Step {step}: Error updating state - {str(e)}")
                yield {
                    'status': 'error',
                    'message': f"Error updating state: {str(e)}",
                    'assignments_dict': assignments_dict
                }
                return

        if step >= max_steps:
            logger.warning(f"Inference terminated after {max_steps} steps without completion")
            yield {
                'status': 'complete',
                'assignments_df': pd.DataFrame(list(assignments_dict.values())),
                'message': f"Reached max steps ({max_steps})"
            }
        else:
            logger.info(f"Inference completed in {step} steps")
            assignments = list(assignments_dict.values())
            logger.info(f"Generated {len(assignments)} assignments")
            yield {
                'status': 'complete',
                'assignments_df': pd.DataFrame(assignments)
            }
            
    except Exception as e:
        logger.error(f"Error in evaluate_model: {str(e)}")
        yield {
            'status': 'error',
            'message': f"Error in evaluate_model: {str(e)}",
            'assignments_dict': assignments_dict
        }