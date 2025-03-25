import numpy as np
import pandas as pd
from google.colab import drive
import gym
from gym import spaces
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)

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

# Subsets
train_emp = employees[:20]
train_tasks = tasks[:50]
train_emp_skills = employee_skills[:20]
train_task_skills = task_skills[:50]
train_story_points = [story_points[t] for t in train_tasks]

test_emp = employees[20:29]
test_tasks = tasks[50:70]
test_emp_skills = employee_skills[20:29]
test_task_skills = task_skills[50:70]
test_story_points = [story_points[t] for t in test_tasks]

# WED function
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

# Environment with Debugging
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
        print(f"Reset: {self.num_tasks} tasks, {self.num_employees} employees, unassigned_tasks: {len(self.unassigned_tasks)}")
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
        print(f"Action: task {task_idx}, emp {emp_idx}")
        valid = (task_idx < self.num_tasks and task_idx in self.unassigned_tasks and
                 emp_idx < self.num_employees and self.workload[emp_idx] + self.story_points[task_idx] <= self.max_workload)
        if valid:
            print(f"Valid action, assigning task {task_idx} to emp {emp_idx}")
            self.assignments[task_idx, emp_idx] = 1
            self.workload[emp_idx] += self.story_points[task_idx]
            self.unassigned_tasks.remove(task_idx)
            reward = self._compute_reward(task_idx, emp_idx)
        else:
            print(f"Invalid action")
            reward = -1
        done = len(self.unassigned_tasks) == 0
        print(f"Unassigned tasks left: {len(self.unassigned_tasks)}, Assignments sum: {np.sum(self.assignments)}")
        return self._get_obs(), reward, done, {}

    def _compute_reward(self, task_idx, emp_idx):
        idle_reward = 1 if self.workload[emp_idx] == 0 else 0.1
        skill_reward = calculate_weighted_euclidean_distance(self.employee_skills[emp_idx], self.task_skills[task_idx])
        temp_workload = self.workload.copy()
        temp_workload[emp_idx] += self.story_points[task_idx]
        active_workloads = temp_workload[temp_workload > 0]
        workload_reward = -np.var(active_workloads) if len(active_workloads) > 1 else 0
        return (0.05 * idle_reward) + (0.85 * skill_reward) + (0.1 * workload_reward)

# Transformer Extractor with Debugging
class TransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, max_tasks=300, max_employees=109):
        feature_dim = observation_space.spaces['sequence'].shape[1]  # 68
        super().__init__(observation_space, features_dim=feature_dim)
        self.max_tasks = max_tasks
        self.max_employees = max_employees
        embedding_dim = 128
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4, dim_feedforward=embedding_dim, batch_first=True),
            num_layers=2
        )

    def forward(self, observations: dict) -> torch.Tensor:
        sequence = observations['sequence']
        print(f"Sequence shape: {sequence.shape}")
        mask = (sequence.sum(dim=-1) != 0).bool()
        embeddings = self.transformer(sequence, src_key_padding_mask=~mask)
        print(f"Embeddings shape: {embeddings.shape}")
        features = embeddings.mean(dim=1)
        print(f"Features shape from extractor: {features.shape}")
        return features

# Transformer Policy with Enhanced Debugging
class TransformerPolicy(MultiInputActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, max_tasks=300, max_employees=109, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=TransformerExtractor,
            features_extractor_kwargs={'max_tasks': max_tasks, 'max_employees': max_employees},
            **kwargs
        )
        self.max_tasks = max_tasks
        self.max_employees = max_employees
        feature_dim = observation_space.spaces['sequence'].shape[1]  # 68
        
        # Shared network with consistent dimensions
        self.shared_net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Action heads with proper dimensions
        self.task_net = nn.Sequential(
            nn.Linear(128, max_tasks)
        )
        
        self.emp_net = nn.Sequential(
            nn.Linear(128, max_employees)
        )
        
        # Value network matching shared features dimension
        self.value_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)  # [batch_size, 68]
        shared_features = self.shared_net(features)  # [batch_size, 128]
        
        # Get logits
        task_logits = self.task_net(shared_features)  # [batch_size, max_tasks]
        emp_logits = self.emp_net(shared_features)    # [batch_size, max_employees]
        
        # Get value
        value = self.value_net(shared_features)       # [batch_size, 1]
        value = value.squeeze(-1)                     # [batch_size]
        
        # Convert masks to proper device and shape
        is_unassigned = obs['is_unassigned'].to(task_logits.device)
        is_available = obs['is_available'].to(emp_logits.device)
        
        task_logits = task_logits.masked_fill(is_unassigned == 0, -1e9)
        emp_logits = emp_logits.masked_fill(is_available == 0, -1e9)
        
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
        return action, value, log_prob

# Train
train_env = MOOEnv(train_task_skills, train_emp_skills, train_story_points)
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
print(f"Value net weight shape: {model.policy.value_net[0].weight.shape}")
model.learn(total_timesteps=50000)
model.save("ppo_transformer_subset_wed")

# Test
print("\nEvaluating trained model...")
test_env = MOOEnv(test_task_skills, test_emp_skills, test_story_points)
loaded_model = PPO.load("ppo_transformer_subset_wed")

obs = test_env.reset()
done = False
step_count = 0
while not done:
    action, _ = loaded_model.predict(obs, deterministic=True)
    obs, reward, done, _ = test_env.step(action)
    step_count += 1
    if step_count > 100:
        print("Breaking after 100 steps to avoid infinite loop")
        break

assignments = test_env.assignments[:len(test_tasks), :len(test_emp)]
active_employees = np.sum(np.any(assignments, axis=0))
similarity_scores = [calculate_weighted_euclidean_distance(test_emp_skills[e], test_task_skills[t]) 
                     for t, e in np.argwhere(assignments)]
workload_var = np.var(test_env.workload[test_env.workload > 0]) if np.any(test_env.workload) else 0

print("Test Results (9 employees x 20 tasks):")
print(f"  Active Employees: {active_employees}/9")
print(f"  Avg Skill Suitability: {np.mean(similarity_scores):.4f}")
print(f"  Workload Variance: {workload_var:.4f}")
print("Assignments:", np.argwhere(assignments))
print("Final unassigned tasks:", test_env.unassigned_tasks)
print("Total assignments made:", np.sum(test_env.assignments))
if len(test_env.unassigned_tasks) == 0 and np.sum(test_env.assignments) == 0:
    print("BUG DETECTED: All tasks 'assigned' but no assignments recorded!")