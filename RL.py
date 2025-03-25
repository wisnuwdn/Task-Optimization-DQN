import numpy as np
import pandas as pd
from google.colab import drive
import gym
from gym import spaces
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy

def mount_drive():
    drive.mount('/content/drive', force_remount=True)

def load_data():
    # Paths
    employee_path = '/content/drive/MyDrive/Skripsi/Resources/Datasets/fixed_data_employee.csv'
    task_path = '/content/drive/MyDrive/Skripsi/Resources/Datasets/fixed_data_task.csv'

    # Load data
    employee_df = pd.read_csv(employee_path, index_col='employee_id').fillna(0)
    employee_df.drop(columns=['No', 'Role'], inplace=True, errors='ignore')
    task_df = pd.read_csv(task_path, index_col='task_id').fillna(0)

    employees = employee_df.index.tolist()
    tasks = task_df.index.tolist()
    story_points = task_df['story_points'].to_dict()
    task_skills = task_df.drop(columns=['project_id', 'story_points']).values
    employee_skills = employee_df.values

    # Create training and test sets
    train_data = {
        'emp': employees[:20],
        'tasks': tasks[:50],
        'emp_skills': employee_skills[:20],
        'task_skills': task_skills[:50],
        'story_points': [story_points[t] for t in tasks[:50]]
    }

    test_data = {
        'emp': employees[20:29],
        'tasks': tasks[50:70],
        'emp_skills': employee_skills[20:29],
        'task_skills': task_skills[50:70],
        'story_points': [story_points[t] for t in tasks[50:70]]
    }

    return train_data, test_data

def calculate_weighted_euclidean_distance(employee_skills, task_skills, alpha=0.05):
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

class MOOEnv(gym.Env):
    def __init__(self, task_skills, employee_skills, story_points, max_workload=20, max_tasks=300, max_employees=109):
        super().__init__()
        self.task_skills = task_skills
        self.employee_skills = employee_skills
        self.num_tasks = task_skills.shape[0]
        self.num_employees = employee_skills.shape[0]
        self.story_points = story_points
        self.max_workload = max_workload
        self.max_tasks = max_tasks
        self.max_employees = max_employees

        self.assignments = np.zeros((self.num_tasks, self.num_employees), dtype=np.float32)
        self.workload = np.zeros(self.num_employees, dtype=np.float32)
        self.unassigned_tasks = list(range(self.num_tasks))

        self.action_space = spaces.MultiDiscrete([max_tasks, max_employees])
        feature_dim = 68
        self.observation_space = spaces.Dict({
            'sequence': spaces.Box(low=-np.inf, high=np.inf, shape=(max_tasks + max_employees, feature_dim), dtype=np.float32),
            'is_unassigned': spaces.Box(low=0, high=1, shape=(max_tasks,), dtype=np.float32),
            'is_available': spaces.Box(low=0, high=1, shape=(max_employees,), dtype=np.float32),
        })

    def reset(self):
        self.assignments = np.zeros((self.num_tasks, self.num_employees))
        self.workload = np.zeros(self.num_employees)
        self.unassigned_tasks = list(range(self.num_tasks))
        return self._get_obs()

    def _get_obs(self):
        sequence = np.zeros((self.max_tasks + self.max_employees, 68), dtype=np.float32)
        for i in range(self.num_tasks):
            sequence[i, :65] = self.task_skills[i]
            sequence[i, 65] = self.story_points[i]
            sequence[i, 66] = 1
            sequence[i, 67] = 1 if i in self.unassigned_tasks else 0
        for j in range(self.num_employees):
            sequence[self.max_tasks + j, :65] = self.employee_skills[j]
            sequence[self.max_tasks + j, 65] = self.workload[j]
            sequence[self.max_tasks + j, 66] = 0
            sequence[self.max_tasks + j, 67] = 1 if self.workload[j] < self.max_workload else 0
        
        is_unassigned = np.zeros(self.max_tasks, dtype=np.float32)
        is_unassigned[:self.num_tasks] = [1 if k in self.unassigned_tasks else 0 for k in range(self.num_tasks)]
        
        is_available = np.zeros(self.max_employees, dtype=np.float32)
        is_available[:self.num_employees] = [1 if self.workload[j] < self.max_workload else 0 for j in range(self.num_employees)]
        
        return {
            'sequence': sequence,
            'is_unassigned': is_unassigned,
            'is_available': is_available,
        }

    def step(self, action):
        task_idx, emp_idx = action
        reward = 0
        done = False
        
        if (task_idx < self.num_tasks and task_idx in self.unassigned_tasks and
            emp_idx < self.num_employees and self.workload[emp_idx] + self.story_points[task_idx] <= self.max_workload):
            self.assignments[task_idx, emp_idx] = 1
            self.workload[emp_idx] += self.story_points[task_idx]
            self.unassigned_tasks.remove(task_idx)
            reward = self._compute_reward(task_idx, emp_idx)
        else:
            reward = -1
            
        done = len(self.unassigned_tasks) == 0
        return self._get_obs(), reward, done, {}

    def _compute_reward(self, task_idx, emp_idx):
        idle_reward = 1 if self.workload[emp_idx] == 0 else 0.1
        skill_reward = calculate_weighted_euclidean_distance(self.employee_skills[emp_idx], self.task_skills[task_idx])
        temp_workload = self.workload.copy()
        temp_workload[emp_idx] += self.story_points[task_idx]
        active_workloads = temp_workload[temp_workload > 0]
        workload_reward = -np.var(active_workloads) if len(active_workloads) > 1 else 0
        return (0.05 * idle_reward) + (0.85 * skill_reward) + (0.1 * workload_reward)

class TransformerPolicy(MultiInputActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, max_tasks=300, max_employees=109, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        self.max_tasks = max_tasks
        self.max_employees = max_employees
        feature_dim = observation_space.spaces['sequence'].shape[1]  # 68
        embedding_dim = 128
        
        # Embedding layer
        self.embedding = nn.Linear(feature_dim, embedding_dim)
        
        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8, dim_feedforward=256, batch_first=True),
            num_layers=2
        )
        
        # Output layers with proper dimensions
        self.task_linear = nn.Linear(embedding_dim, 1)
        self.emp_linear = nn.Linear(embedding_dim, 1)
        
        # Modified value network
        self.value_net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),  # Keep dimensions consistent
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, obs, deterministic=False):
        sequence = obs['sequence']  # [batch, 409, 68]
        is_unassigned = obs['is_unassigned']  # [batch, 300]
        is_available = obs['is_available']  # [batch, 109]
        
        # Ensure batch dimension and convert to float
        if len(sequence.shape) == 2:
            sequence = sequence.unsqueeze(0)
        sequence = sequence.float()
        
        # Project to embedding space
        sequence = self.embedding(sequence)  # [batch, 409, 128]
        
        # Create attention mask
        mask = (obs['sequence'].sum(dim=-1) != 0).bool()
        if len(mask.shape) == 1:
            mask = mask.unsqueeze(0)
        
        # Apply transformer
        embeddings = self.transformer(sequence, src_key_padding_mask=~mask)  # [batch, 409, 128]
        
        # Split embeddings
        task_embeddings = embeddings[:, :self.max_tasks, :]  # [batch, 300, 128]
        emp_embeddings = embeddings[:, self.max_tasks:self.max_tasks + self.max_employees, :]  # [batch, 109, 128]
        
        # Generate logits
        task_logits = self.task_linear(task_embeddings).squeeze(-1)  # [batch, 300]
        emp_logits = self.emp_linear(emp_embeddings).squeeze(-1)  # [batch, 109]
        
        # Ensure proper dimensions for masks
        if len(is_unassigned.shape) == 1:
            is_unassigned = is_unassigned.unsqueeze(0)
        if len(is_available.shape) == 1:
            is_available = is_available.unsqueeze(0)
            
        # Mask invalid actions
        task_logits = task_logits.masked_fill(is_unassigned == 0, -1e9)
        emp_logits = emp_logits.masked_fill(is_available == 0, -1e9)
        
        # Action distributions
        task_dist = torch.distributions.Categorical(logits=task_logits)
        emp_dist = torch.distributions.Categorical(logits=emp_logits)
        
        if deterministic:
            task_action = task_dist.probs.argmax(dim=-1)
            emp_action = emp_dist.probs.argmax(dim=-1)
        else:
            task_action = task_dist.sample()
            emp_action = emp_dist.sample()
        
        action = torch.stack([task_action, emp_action], dim=1)
        log_prob = task_dist.log_prob(task_action) + emp_dist.log_prob(emp_action)
        
        # Value head with proper reshaping
        # Modified value computation
        value_input = embeddings.mean(dim=1)  # [batch, 128]
        value = self.value_net(value_input)  # [batch, 1]
        
        return action, value, log_prob

def main():
    # Mount drive and load data
    mount_drive()
    train_data, test_data = load_data()
    
    # Print shapes for debugging
    print("Training Data Shapes:")
    print(f"Task Skills Shape: {train_data['task_skills'].shape}")
    print(f"Employee Skills Shape: {train_data['emp_skills'].shape}")
    print(f"Story Points Length: {len(train_data['story_points'])}")
    
    # Create and initialize environment
    train_env = MOOEnv(
        train_data['task_skills'],
        train_data['emp_skills'],
        train_data['story_points']
    )
    
    # Print initial observation shapes
    obs = train_env.reset()
    print("\nObservation Shapes:")
    for key, value in obs.items():
        print(f"{key}: {value.shape}")
    
    # Initialize and train model
    model = PPO(
        TransformerPolicy,
        train_env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        policy_kwargs={'max_tasks': 300, 'max_employees': 109},
        learning_rate=3e-4
    )
    
    # Train
    model.learn(total_timesteps=50000)
    model.save("ppo_transformer_subset_wed")

if __name__ == "__main__":
    main()