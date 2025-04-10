import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv
from sklearn.impute import SimpleImputer
import gym
from gym import spaces
from collections import deque
import random
from google.colab import drive
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

# Mount Google Drive
drive.mount('/content/drive')

# Paths
task_data_path = '/content/drive/MyDrive/Skripsi/Resources/Datasets/fixed_data_task.csv'
employee_data_path = '/content/drive/MyDrive/Skripsi/Resources/Datasets/fixed_data_employee.csv'
model_save_path = '/content/drive/MyDrive/Skripsi/dqn_model.pth'
agent_save_path = '/content/drive/MyDrive/Skripsi/dqn_agent_params.pth'

# Load and clean datasets
task_data = pd.read_csv(task_data_path)
employee_data = pd.read_csv(employee_data_path)
task_subset = task_data.iloc[:300]
employee_subset = employee_data.iloc[:109]
skill_columns = [col for col in task_data.columns if col not in ['task_id', 'story_points', 'project_id']]
task_skills = task_subset[skill_columns].values
employee_skills = employee_subset[skill_columns].values
story_points = task_subset['story_points'].values

# Normalize and impute
task_skills = task_skills / 5.0
employee_skills = employee_skills / 5.0
imputer = SimpleImputer(strategy='mean')
task_skills = imputer.fit_transform(task_skills)
employee_skills = imputer.fit_transform(employee_skills)

# Define maximum sizes
max_tasks = 300
max_employees = 109

# Task Assignment Environment
class TaskAssignmentEnv(gym.Env):
    def __init__(self, task_skills, employee_skills, story_points, max_workload=20, max_tasks=max_tasks, max_employees=max_employees):
        super(TaskAssignmentEnv, self).__init__()
        self.task_skills = task_skills
        self.employee_skills = employee_skills
        self.story_points = story_points
        self.max_workload = max_workload
        self.max_tasks = max_tasks
        self.max_employees = max_employees
        self.num_tasks = len(task_skills)
        self.num_employees = len(employee_skills)
        self.task_feature_dim = 66
        self.employee_feature_dim = 66
        self.observation_space = spaces.Dict({
            'task_features': spaces.Box(low=0, high=1, shape=(self.max_tasks, self.task_feature_dim), dtype=np.float32),
            'employee_features': spaces.Box(low=0, high=1, shape=(self.max_employees, self.employee_feature_dim), dtype=np.float32),
            'task_mask': spaces.Box(low=0, high=1, shape=(self.max_tasks,), dtype=np.float32)
        })
        self.action_space = spaces.MultiDiscrete([self.max_tasks, self.max_employees])
        self.edge_index = self._create_edge_index()

    def _create_edge_index(self):
        edge_index = []
        for t in range(self.max_tasks):
            for e in range(self.max_employees):
                edge_index.append([t, self.max_tasks + e])
                edge_index.append([self.max_tasks + e, t])
        return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    def reset(self):
        self.workloads = np.zeros(self.num_employees)
        self.assignments = -np.ones(self.num_tasks, dtype=int)
        self.remaining_tasks = list(range(self.num_tasks))
        self.wed_scores = []
        self.similarity_scores = []
        return self._get_obs()

    def _get_obs(self):
        task_features = np.zeros((self.max_tasks, self.task_feature_dim))
        for t in range(self.num_tasks):
            task_features[t, :65] = self.task_skills[t]
            task_features[t, 65] = self.story_points[t] / 20.0

        employee_features = np.zeros((self.max_employees, self.employee_feature_dim))
        for e in range(self.num_employees):
            employee_features[e, :65] = self.employee_skills[e]
            employee_features[e, 65] = self.workloads[e] / self.max_workload

        task_mask = np.zeros(self.max_tasks)
        for t in range(self.num_tasks):
            task_mask[t] = 1 if t in self.remaining_tasks else 0

        return {
            'task_features': task_features.astype(np.float32),
            'employee_features': employee_features.astype(np.float32),
            'task_mask': task_mask.astype(np.float32)
        }

    def _compute_wed(self, emp_idx, task_idx):
        emp_skills = self.employee_skills[emp_idx]
        task_skills = self.task_skills[task_idx]
        diff = emp_skills - task_skills
        wed = np.sqrt(np.sum(diff ** 2))
        if np.isnan(wed):
            print(f"Warning: WED is nan for emp_idx={emp_idx}, task_idx={task_idx}")
        return wed

    def step(self, action):
        task_idx, emp_idx = action
        done = False
        reward = 0
        info = {}

        if task_idx >= self.num_tasks or emp_idx >= self.num_employees:
            reward = -1
        elif task_idx not in self.remaining_tasks:
            reward = -1
        elif self.workloads[emp_idx] + self.story_points[task_idx] > self.max_workload:
            reward = -1
        else:
            self.assignments[task_idx] = emp_idx
            self.workloads[emp_idx] += self.story_points[task_idx]
            self.remaining_tasks.remove(task_idx)
            wed = self._compute_wed(emp_idx, task_idx)
            self.wed_scores.append(wed)
            similarity = 1 - (wed / np.sqrt(65 * 25))
            if np.isnan(similarity):
                print(f"Error: Similarity is nan for task_idx={task_idx}, emp_idx={emp_idx}, wed={wed}")
                reward = 0
            else:
                reward = similarity
            self.similarity_scores.append(similarity)

        if len(self.remaining_tasks) == 0:
            done = True
            std_workloads = np.std(self.workloads)
            num_idle = np.sum(self.workloads == 0)
            penalty = -0.5 * (std_workloads / self.max_workload) - 0.5 * (num_idle / self.num_employees)
            reward += penalty
            info = {'total_wed': np.sum(self.wed_scores), 'similarity_scores': self.similarity_scores}

        return self._get_obs(), reward, done, info

# GNN and DQN Models
class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)
        self.conv2 = GATConv(hidden_dim * 4, output_dim, heads=1, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class DQN(nn.Module):
    def __init__(self, gnn, hidden_dim, max_tasks, max_employees):
        super(DQN, self).__init__()
        self.gnn = gnn
        self.max_tasks = max_tasks
        self.max_employees = max_employees
        self.embedding_dim = 32
        self.fc1 = nn.Linear(self.embedding_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, obs, graph_data, valid_action_indices=None):
        embeddings = self.gnn(graph_data)
        task_embeddings = embeddings[:self.max_tasks]
        employee_embeddings = embeddings[self.max_tasks:]

        if valid_action_indices is not None:
            # Only compute Q-values for valid actions
            task_indices = valid_action_indices[:, 0]
            emp_indices = valid_action_indices[:, 1]
            task_emb = task_embeddings[task_indices]  # (num_valid, embedding_dim)
            emp_emb = employee_embeddings[emp_indices]  # (num_valid, embedding_dim)
            pair_embeddings = torch.cat([task_emb, emp_emb], dim=-1)  # (num_valid, 2*embedding_dim)
            x = F.relu(self.fc1(pair_embeddings))
            q_values = self.fc2(x).squeeze(-1)  # (num_valid,)
            return q_values
        else:
            # Compute Q-values for all actions
            task_embeddings = task_embeddings.unsqueeze(1).expand(-1, self.max_employees, -1)
            employee_embeddings = employee_embeddings.unsqueeze(0).expand(self.max_tasks, -1, -1)
            pair_embeddings = torch.cat([task_embeddings, employee_embeddings], dim=-1)
            x = F.relu(self.fc1(pair_embeddings))
            q_values = self.fc2(x).squeeze(-1)
            return q_values

# DQN Agent
class DQNAgent:
    def __init__(self, task_feature_dim, employee_feature_dim, max_tasks, max_employees):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gnn = GNN(input_dim=66, hidden_dim=16, output_dim=32).to(self.device)
        self.model = DQN(self.gnn, hidden_dim=128, max_tasks=max_tasks, max_employees=max_employees).to(self.device)
        self.target_model = DQN(self.gnn, hidden_dim=128, max_tasks=max_tasks, max_employees=max_employees).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        self.memory = deque(maxlen=50000)
        self.batch_size = 64
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        self.max_tasks = max_tasks
        self.max_employees = max_employees

    def act(self, obs, graph_data, env):
        task_mask = torch.tensor(obs['task_mask'], dtype=torch.float32).to(self.device)
        # Compute valid employees based on workload
        employee_mask = torch.ones(env.max_employees, device=self.device)
        for e in range(env.num_employees):
            current_workload = obs['employee_features'][e, 65] * env.max_workload
            # Use the story point of the first unassigned task (simplification)
            next_task_idx = next(t for t in range(env.num_tasks) if task_mask[t] == 1)
            if current_workload + env.story_points[next_task_idx] > env.max_workload:
                employee_mask[e] = 0
        employee_mask[env.num_employees:] = 0

        # Get valid actions
        valid_tasks = [t for t in range(env.num_tasks) if task_mask[t] == 1]
        valid_employees = [e for e in range(env.max_employees) if employee_mask[e] == 1]
        if not valid_tasks or not valid_employees:
            return [0, 0]

        # Create valid action indices
        valid_action_indices = []
        for t in valid_tasks:
            for e in valid_employees:
                valid_action_indices.append([t, e])
        valid_action_indices = torch.tensor(valid_action_indices, dtype=torch.long).to(self.device)

        if random.random() < self.epsilon:
            action_idx = random.randint(0, len(valid_action_indices) - 1)
            action = valid_action_indices[action_idx]
            return action.tolist()
        else:
            with torch.no_grad():
                q_values = self.model(obs, graph_data, valid_action_indices=valid_action_indices)
                action_idx = torch.argmax(q_values).item()
                action = valid_action_indices[action_idx]
                return action.tolist()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, env):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Create batched graphs
        state_graphs = [Data(x=torch.cat([torch.tensor(s['task_features'], dtype=torch.float32),
                                          torch.tensor(s['employee_features'], dtype=torch.float32)]),
                             edge_index=env.edge_index) for s in states]
        state_batch = Batch.from_data_list(state_graphs).to(self.device)

        next_state_graphs = [Data(x=torch.cat([torch.tensor(s['task_features'], dtype=torch.float32),
                                               torch.tensor(s['employee_features'], dtype=torch.float32)]),
                                  edge_index=env.edge_index) for s in next_states]
        next_state_batch = Batch.from_data_list(next_state_graphs).to(self.device)

        # Compute embeddings
        state_embeddings = self.model.gnn(state_batch)
        next_state_embeddings = self.target_model.gnn(next_state_batch)

        # Split embeddings
        state_task_embeddings = state_embeddings[:self.batch_size * self.max_tasks].view(self.batch_size, self.max_tasks, -1)
        state_employee_embeddings = state_embeddings[self.batch_size * self.max_tasks:].view(self.batch_size, self.max_employees, -1)
        next_state_task_embeddings = next_state_embeddings[:self.batch_size * self.max_tasks].view(self.batch_size, self.max_tasks, -1)
        next_state_employee_embeddings = next_state_embeddings[self.batch_size * self.max_tasks:].view(self.batch_size, self.max_employees, -1)

        # Compute Q-values
        q_values = []
        next_q_values = []
        for i in range(self.batch_size):
            task_emb = state_task_embeddings[i]
            emp_emb = state_employee_embeddings[i]
            pair_emb = torch.cat([task_emb.unsqueeze(1).expand(-1, self.max_employees, -1),
                                  emp_emb.unsqueeze(0).expand(self.max_tasks, -1, -1)], dim=-1)
            x = F.relu(self.model.fc1(pair_emb))
            q_vals = self.model.fc2(x).squeeze(-1)
            q_values.append(q_vals)

            with torch.no_grad():
                next_task_emb = next_state_task_embeddings[i]
                next_emp_emb = next_state_employee_embeddings[i]
                next_pair_emb = torch.cat([next_task_emb.unsqueeze(1).expand(-1, self.max_employees, -1),
                                           next_emp_emb.unsqueeze(0).expand(self.max_tasks, -1, -1)], dim=-1)
                next_x = F.relu(self.target_model.fc1(next_pair_emb))
                next_q_vals = self.target_model.fc2(next_x).squeeze(-1)
                next_q_values.append(next_q_vals)

        q_values = torch.stack(q_values)
        next_q_values = torch.stack(next_q_values)

        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        task_idx = actions[:, 0]
        emp_idx = actions[:, 1]
        q_values_selected = q_values[torch.arange(self.batch_size), task_idx, emp_idx]

        next_q_values_max = next_q_values.max(dim=2)[0].max(dim=1)[0]
        targets = rewards + (1 - dones) * self.gamma * next_q_values_max

        loss = F.mse_loss(q_values_selected, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, model_path, agent_path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, model_path)
        torch.save({'epsilon': self.epsilon, 'memory': list(self.memory)}, agent_path)
        print(f"Model saved to {model_path}, Agent params to {agent_path}")

    def load(self, model_path, agent_path):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent_params = torch.load(agent_path)
        self.epsilon = agent_params['epsilon']
        self.memory = deque(agent_params['memory'], maxlen=50000)
        print(f"Model loaded from {model_path}, Agent params from {agent_path}")

# Training Loop
def train_dqn():
    env = TaskAssignmentEnv(task_skills, employee_skills, story_points)
    agent = DQNAgent(env.task_feature_dim, env.employee_feature_dim, max_tasks, max_employees)
    num_episodes = 10
    rewards = []
    similarity_scores_list = []

    for episode in range(num_episodes):
        obs = env.reset()
        graph_data = Data(x=torch.cat([torch.tensor(obs['task_features']), torch.tensor(obs['employee_features'])]).to(agent.device),
                         edge_index=env.edge_index.to(agent.device))
        total_reward = 0
        done = False
        step = 0

        print(f"Episode {episode+1}/{num_episodes}")
        while not done:
            step += 1
            start_time = time.time()
            action = agent.act(obs, graph_data, env)
            next_obs, reward, done, info = env.step(action)
            next_graph_data = Data(x=torch.cat([torch.tensor(next_obs['task_features']), torch.tensor(next_obs['employee_features'])]).to(agent.device),
                                  edge_index=env.edge_index.to(agent.device))
            agent.remember(obs, action, reward, next_obs, done)
            agent.replay(env)
            obs = next_obs
            graph_data = next_graph_data
            total_reward += reward
            step_time = time.time() - start_time
            print(f"Step {step}/{env.num_tasks}: Action {action}, Reward {reward:.4f}, Time {step_time:.2f}s")

        agent.update_target()
        rewards.append(total_reward)
        if 'similarity_scores' in info:
            similarity_scores_list.append(info['similarity_scores'])
        print(f"Episode {episode+1} - Total Reward: {total_reward:.2f}")

    # Save the model
    agent.save(model_save_path, agent_save_path)

    # Visualization
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_episodes + 1), rewards, marker='o')
        plt.title('Total Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        plt.show()

        if similarity_scores_list:
            plt.figure(figsize=(10, 5))
            sns.boxplot(data=similarity_scores_list[-1])
            plt.title('Similarity Scores in Last Episode')
            plt.ylabel('Similarity Score')
            plt.show()
    except Exception as e:
        print(f"Visualization failed: {e}. Model is still saved.")

# Run training
train_dqn()