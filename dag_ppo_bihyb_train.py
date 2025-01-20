import argparse
import torch
from torch import nn
import os
import time
import yaml
import subprocess
import sys
import random
import numpy as np
from torch.multiprocessing import Pool, cpu_count
from copy import deepcopy
import psutil
from src.dag_ppo_bihyb_model import ActorNet, CriticNet, GraphEncoder
from utils.utils import print_args
from utils.tfboard_helper import TensorboardUtil
from utils.dag_graph import DAGraph
from dag_data.dag_generator import load_tpch_tuples
from dag_ppo_bihyb_eval import evaluate, evaluate_gfn
import psutil
from collections import deque

class ItemsContainer:
    def __init__(self):
        self.__reward = []
        self.__inp_graph = []
        self.__ori_greedy = []
        self.__greedy = []
        # self.__edge_candidates = []
        self.__forward_edge_candidates = []
        self.__backward_edge_candidates = []
        self.__done = []

    # def append(self, reward, inp_graph, greedy, edge_candidates, done, ori_greedy):
    def append(self, reward, inp_graph, greedy, forward_edge_candidates, backward_edge_candidates, done, ori_greedy):
        self.__reward.append(reward)
        self.__inp_graph.append(inp_graph)
        self.__greedy.append(greedy)
        # self.__edge_candidates.append(edge_candidates)
        self.__forward_edge_candidates.append(forward_edge_candidates)
        self.__backward_edge_candidates.append(backward_edge_candidates)
        self.__done.append(done)
        self.__ori_greedy.append(ori_greedy)

    @property
    def reward(self):
        return deepcopy(self.__reward)

    @property
    def inp_graph(self):
        return deepcopy(self.__inp_graph)

    @property
    def greedy(self):
        return deepcopy(self.__greedy)

    # @property
    # def edge_candidates(self):
    #     return deepcopy(self.__edge_candidates)
    
    @property
    def forward_edge_candidates(self):
        return deepcopy(self.__forward_edge_candidates)
    
    @property
    def backward_edge_candidates(self):
        return deepcopy(self.__backward_edge_candidates)

    @property
    def done(self):
        return deepcopy(self.__done)

    @property
    def ori_greedy(self):
        return deepcopy(self.__ori_greedy)

    # def update(self, idx, reward=None, inp_graph=None, greedy=None, edge_candidates=None, done=None, ori_greedy=None):
    def update(self, idx, reward=None, inp_graph=None, greedy=None, forward_edge_candidates=None, backward_edge_candidates=None, done=None, ori_greedy=None):
        if reward is not None:
            self.__reward[idx] = reward
        if inp_graph is not None:
            self.__inp_graph[idx] = inp_graph
        if greedy is not None:
            self.__greedy[idx] = greedy
        # if edge_candidates is not None:
        #     self.__edge_candidates[idx] = edge_candidates
        if forward_edge_candidates is not None:
            self.__forward_edge_candidates[idx] = forward_edge_candidates
        if backward_edge_candidates is not None:
            self.__backward_edge_candidates[idx] = backward_edge_candidates
        if done is not None:
            self.__done[idx] = done
        if ori_greedy is not None:
            self.__ori_greedy[idx] = ori_greedy

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.next_states = []
        # self.candidates = []
        self.forward_candidates = []
        self.backward_candidates = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.next_states[:]
        # del self.candidates[:]
        del self.forward_candidates[:]
        del self.backward_candidates[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class Memory_deque:
    def __init__(self, maxlen=100):  # maxlen 기본값 설정
        self.actions = deque(maxlen=maxlen)
        self.states = deque(maxlen=maxlen)
        self.next_states = deque(maxlen=maxlen)
        # self.candidates = deque(maxlen=maxlen)
        self.forward_candidates = deque(maxlen=maxlen)
        self.backward_candidates = deque(maxlen=maxlen)
        self.logprobs = deque(maxlen=maxlen)
        self.rewards = deque(maxlen=maxlen)
        self.is_terminals = deque(maxlen=maxlen)

    def clear_memory(self):
        self.actions.clear()
        self.states.clear()
        self.next_states.clear()
        # self.candidates.clear()
        self.forward_candidates.clear()
        self.backward_candidates.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()
    
    def merge(self, other_memory):
            """
            Merges another Memory_deque into this one.
            If the combined size exceeds maxlen, the oldest entries will be removed.
            """
            self.actions.extend(other_memory.actions)
            self.states.extend(other_memory.states)
            self.next_states.extend(other_memory.next_states)
            # self.candidates.extend(other_memory.candidates)
            self.forward_candidates.extend(other_memory.forward_candidates)
            self.backward_candidates.extend(other_memory.backward_candidates)
            self.logprobs.extend(other_memory.logprobs)
            self.rewards.extend(other_memory.rewards)
            self.is_terminals.extend(other_memory.is_terminals)

    def trim_recent(self, sample_size):
        """
        Sample the most recent `sample_size` items and trim the memory.
        Returns a new Memory_deque containing the sampled items.
        """
        # 기존 memory에서 잘라내기
        self.actions = deque(list(self.actions)[:-sample_size], maxlen=self.actions.maxlen)
        self.states = deque(list(self.states)[:-sample_size], maxlen=self.states.maxlen)
        self.next_states = deque(list(self.next_states)[:-sample_size], maxlen=self.next_states.maxlen)
        # self.candidates = deque(list(self.candidates)[:-sample_size], maxlen=self.candidates.maxlen)
        self.forward_candidates = deque(list(self.forward_candidates)[:-sample_size], maxlen=self.forward_candidates.maxlen)
        self.backward_candidates = deque(list(self.backward_candidates)[:-sample_size], maxlen=self.backward_candidates.maxlen)
        self.logprobs = deque(list(self.logprobs)[:-sample_size], maxlen=self.logprobs.maxlen)
        self.rewards = deque(list(self.rewards)[:-sample_size], maxlen=self.rewards.maxlen)
        self.is_terminals = deque(list(self.is_terminals)[:-sample_size], maxlen=self.is_terminals.maxlen)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # 안정성을 위해 최대값을 빼줌 (overflow 방지)
    return exp_x / exp_x.sum()

def sample_memory(memory, sample_size=20):
    
    rewards = np.array(memory.rewards, dtype=np.float32)
    probabilities = softmax(rewards)
    probabilities_tensor = torch.tensor(probabilities, dtype=torch.float32).squeeze()
    
    sampled_indices = torch.multinomial(probabilities_tensor, num_samples=sample_size, replacement=False)
    sampled_memory = Memory_deque(maxlen=sample_size)

    # 샘플링된 인덱스에 따라 각 속성 복사
    for idx in sampled_indices:
        sampled_memory.actions.append(memory.actions[idx])
        sampled_memory.states.append(memory.states[idx])
        sampled_memory.next_states.append(memory.next_states[idx])
        # sampled_memory.candidates.append(memory.candidates[idx])
        sampled_memory.forward_candidates.append(memory.forward_candidates[idx])
        sampled_memory.backward_candidates.append(memory.backward_candidates[idx])
        sampled_memory.logprobs.append(memory.logprobs[idx])
        sampled_memory.rewards.append(memory.rewards[idx])
        sampled_memory.is_terminals.append(memory.is_terminals[idx])

    sampled_rewards = np.array(sampled_memory.rewards, dtype=np.float32)
    # print(sampled_rewards)
    return sampled_memory

def filter_memory(memory, threshold=-0.0):
    """
    Filters the memory to remove items with rewards <= threshold.
    The original memory is updated to only contain the remaining items.
    """
    rewards = np.array(memory.rewards, dtype=np.float32)
    valid_indices = np.where(rewards > threshold)[0]

    memory.actions = deque([memory.actions[i] for i in valid_indices], maxlen=memory.actions.maxlen)
    memory.states = deque([memory.states[i] for i in valid_indices], maxlen=memory.states.maxlen)
    memory.next_states = deque([memory.next_states[i] for i in valid_indices], maxlen=memory.next_states.maxlen)
    # memory.candidates = deque([memory.candidates[i] for i in valid_indices], maxlen=memory.candidates.maxlen)
    memory.forward_candidates = deque([memory.forward_candidates[i] for i in valid_indices], maxlen=memory.forward_candidates.maxlen)
    memory.backward_candidates = deque([memory.backward_candidates[i] for i in valid_indices], maxlen=memory.backward_candidates.maxlen)
    memory.logprobs = deque([memory.logprobs[i] for i in valid_indices], maxlen=memory.logprobs.maxlen)
    memory.rewards = deque([memory.rewards[i] for i in valid_indices], maxlen=memory.rewards.maxlen)
    memory.is_terminals = deque([memory.is_terminals[i] for i in valid_indices], maxlen=memory.is_terminals.maxlen)

    return memory

def sample_recent_memory(memory, sample_size=20):
    # 최근 `sample_size`개의 데이터를 추출
    sampled_memory = Memory_deque(maxlen=sample_size)

    # 슬라이싱으로 최근 데이터 추출
    sampled_memory.actions.extend(list(memory.actions)[-sample_size:])
    sampled_memory.states.extend(list(memory.states)[-sample_size:])
    sampled_memory.next_states.extend(list(memory.next_states)[-sample_size:])
    # sampled_memory.candidates.extend(list(memory.candidates)[-sample_size:])
    sampled_memory.forward_candidates.extend(list(memory.forward_candidates)[-sample_size:])
    sampled_memory.backward_candidates.extend(list(memory.backward_candidates)[-sample_size:])
    sampled_memory.logprobs.extend(list(memory.logprobs)[-sample_size:])
    sampled_memory.rewards.extend(list(memory.rewards)[-sample_size:])
    sampled_memory.is_terminals.extend(list(memory.is_terminals)[-sample_size:])

    return sampled_memory

class ActorCritic(nn.Module):
    def __init__(self, dag_graph, node_feature_dim, node_output_size, batch_norm, one_hot_degree, gnn_layers):
        super(ActorCritic, self).__init__()

        self.state_encoder = GraphEncoder(node_feature_dim, node_output_size, batch_norm, one_hot_degree, gnn_layers)
        self.actor_net = ActorNet(dag_graph, node_output_size * 4, batch_norm)
        self.value_net = CriticNet(dag_graph, node_output_size * 4, batch_norm)

    def forward(self):
        raise NotImplementedError

    # def act(self, inp_graph, edge_candidates, memory):
    def act(self, inp_graph, forward_edge_candidates, backward_edge_candidates, memory):
        state_feat = self.state_encoder(inp_graph)
        # actions, action_logits, entropy = self.actor_net(state_feat, edge_candidates)
        actions, action_logits, entropy = self.actor_net(state_feat, forward_edge_candidates)

        memory.states.append(inp_graph)
        # memory.candidates.append(edge_candidates)
        memory.forward_candidates.append(forward_edge_candidates)
        memory.backward_candidates.append(backward_edge_candidates)
        memory.actions.append(actions)
        memory.logprobs.append(action_logits)

        return actions

    def evaluate(self, inp_graph, edge_candidates, action):
        state_feat = self.state_encoder(inp_graph)
        _, action_logits, entropy = self.actor_net(state_feat, edge_candidates, action)
        state_value = self.value_net(state_feat)

        return action_logits, state_value, entropy


class PPO:
    def __init__(self, dag_graph, args, device):
        self.lr = args.learning_rate
        self.betas = args.betas
        self.gamma = args.gamma
        self.eps_clip = args.eps_clip
        self.K_epochs = args.k_epochs

        self.device = device

        ac_params = dag_graph, args.node_feature_dim, args.node_output_size, args.batch_norm, \
                    args.one_hot_degree, args.gnn_layers

        self.policy = ActorCritic(*ac_params).to(self.device)
        self.optimizer = torch.optim.Adam(
            [{'params': self.policy.actor_net.parameters()},
             {'params': self.policy.value_net.parameters()},
             {'params': self.policy.state_encoder.parameters(), 'lr': self.lr / 10}],
            lr=self.lr, betas=self.betas)
        if len(args.lr_steps) > 0:
            # rescale lr_step value to match the action steps
            lr_steps = [step // args.update_timestep for step in args.lr_steps]
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, lr_steps, gamma=0.1)
        else:
            self.lr_scheduler = None
        self.policy_old = ActorCritic(*ac_params).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Time Difference estimate of state rewards:
        rewards = []

        with torch.no_grad():
            logprobs, state_values, dist_entropy = \
                self.policy.evaluate(memory.states[-1], memory.forward_candidates[-1], memory.actions[-1].to(self.device))
        discounted_reward = state_values

        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
            discounted_reward = discounted_reward * (1 - torch.tensor(is_terminal, dtype=torch.float32).to(self.device))
            discounted_reward = reward + (self.gamma * discounted_reward).clone()
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.cat(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = []
        for state in memory.states:
            old_states += state
        old_actions = torch.cat(memory.actions, dim=1)
        old_logprobs = torch.cat(memory.logprobs, dim=1)
        old_candidates = []
        # for candi in memory.candidates:
        for candi in memory.forward_candidates:
            old_candidates += candi

        critic_loss_sum = 0

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_candidates, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs)

            # Normalizing advantages
            advantages = rewards - state_values.detach()
            #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            # Finding Surrogate Loss:
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2)
            critic_loss = self.MseLoss(state_values, rewards)
            entropy_reg = -0.01 * dist_entropy
            critic_loss_sum += critic_loss.detach().mean()

            # take gradient step
            self.optimizer.zero_grad()
            (actor_loss + critic_loss + entropy_reg).mean().backward()
            self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

        return critic_loss_sum / self.K_epochs  # mean critic loss


class GFN(nn.Module):
    def __init__(self, dag_graph, args, device):
        super(GFN, self).__init__()
        self.lr = args.learning_rate
        self.betas = args.betas
        # self.gamma = args.gamma
        # self.eps_clip = args.eps_clip
        # self.K_epochs = args.k_epochs
        
        self.device = device
        
        # ac_params = dag_graph, args.node_feature_dim, args.node_output_size, args.batch_norm, \
        #             args.one_hot_degree, args.gnn_layers
                    
        self.state_encoder = GraphEncoder(args.node_feature_dim, args.node_output_size, args.batch_norm, args.one_hot_degree, args.gnn_layers)
        self.forward_policy = ActorNet(dag_graph, args.node_output_size * 4, args.batch_norm)
        self.flow_model = CriticNet(dag_graph, args.node_output_size * 4, args.batch_norm)

        self.state_encoder.to(self.device)
        self.forward_policy.to(self.device)
        self.flow_model.to(self.device)

        self.optimizer = torch.optim.Adam(
            [{'params': self.forward_policy.parameters()},
             {'params': self.flow_model.parameters(), 'lr': self.lr * 10}, # we usually assign high learning rate for flow model
             {'params': self.state_encoder.parameters(), 'lr': self.lr / 10}],
            lr=self.lr, betas=self.betas)
        if len(args.lr_steps) > 0:
            # rescale lr_step value to match the action steps
            lr_steps = [step // args.update_timestep for step in args.lr_steps]
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, lr_steps, gamma=0.1)
        else:
            self.lr_scheduler = None
            
        self.MseLoss = nn.MSELoss()
        
    # def act(self, inp_graph, edge_candidates, memory):
    def act(self, inp_graph, forward_edge_candidates, backward_edge_candidates, memory):
        state_feat = self.state_encoder(inp_graph)
        # actions, action_logits, entropy = self.forward_policy(state_feat, edge_candidates)
        actions, action_logits, entropy = self.forward_policy(state_feat, forward_edge_candidates)
        
        memory.states.append(inp_graph)
        # memory.candidates.append(edge_candidates)
        memory.forward_candidates.append(forward_edge_candidates)
        memory.backward_candidates.append(backward_edge_candidates)
        memory.actions.append(actions)
        memory.logprobs.append(action_logits)
        
        return actions
    
    def evaluate(self, inp_graph, edge_candidates, action):
        state_feat = self.state_encoder(inp_graph)
        _, action_logits, _ = self.forward_policy(state_feat, edge_candidates, action)
        # state_value = self.flow_model(state_feat)
        
        return action_logits
        
    def update(self, memory):
        states = []
        for state in memory.states:
            states += state
        

        actions = torch.cat(list(memory.actions), dim=1)
        rewards = torch.tensor(list(memory.rewards), dtype=torch.float32).to(self.device).flatten()
        next_states = []
        for state in list(memory.next_states):
            next_states += state
        forward_candidates = []
        for candi in list(memory.forward_candidates):
            forward_candidates += candi
        backward_candidates = []
        for candi in list(memory.backward_candidates):
            backward_candidates += candi
        
        # log p(a1, a2) = log p(a1) + log p(a2|a1)
        log_pf = self.evaluate(states, forward_candidates, actions).sum(dim=0)
        # we set pb as uniform policy. pb = 1/(number of possible backward actions from current state)
        # log_pb = torch.zeros_like(log_pf).to(self.device)
        log_pb = torch.tensor([1/(sum([len(v) for k, v in candi.items()]) // 2 + 1e-7) for candi in backward_candidates]).to(self.device).log()
        state_feats = self.state_encoder(states)
        log_fs = self.flow_model(state_feats)

        total_loss = torch.zeros((len(states)-1)).to(self.device)
        total_loss += log_fs[:-1]
        total_loss += log_pf[:-1]
        total_loss -= log_pb[1:]
        total_loss -= log_fs[1:]
        # FL-DB parametrization. We consider reward as intermediate energy function over transitions
        total_loss -= rewards[:-1]
        total_loss = total_loss.pow(2).mean()
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()
            
        return total_loss

def main(args):
    # initialize manual seed
    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    # create DAG graph environment
    resource_dim = 1
    raw_node_feature_dim = 1 + resource_dim  # (duration, resources)
    args.node_feature_dim = raw_node_feature_dim
    dag_graph = DAGraph(resource_dim=resource_dim,
                        feature_dim=args.node_feature_dim,
                        scheduler_type=args.scheduler_type)

    # load training/testing data
    vargs = (
        dag_graph,
        args.num_init_dags,
        raw_node_feature_dim,
        resource_dim,
        args.resource_limit,
        args.add_graph_features,
        args.scheduler_type
    )
    tuples_train, tuples_test = \
        load_tpch_tuples(args.train_sample, 0, *vargs), load_tpch_tuples(args.test_sample, 1, *vargs)

    # create tensorboard summary writer
    try:
        import tensorflow as tf
        # local mode: logs stored in ./runs/TIME_STAMP-MACHINE_ID
        tfboard_path = 'runs'
        import socket
        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        tfboard_path = os.path.join(tfboard_path, current_time + '_' + socket.gethostname())
        # summary_writer = TensorboardUtil(tf.summary.FileWriter(tfboard_path))
        summary_writer = tf.summary.create_file_writer(tfboard_path)        
    except (ModuleNotFoundError, ImportError):
        print('Warning: Tensorboard not loading, please install tensorflow to enable...')
        summary_writer = None

    # get current device (cuda or cpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # init models
    memory = Memory()
    # memory =  [Memory_deque(maxlen=300) for i in range(args.num_init_dags)]
    # memory = {} 
    # memory =  Memory_deque(maxlen=300) 
    if args.model == 'ppo':
        ppo = PPO(dag_graph, args, device)
        num_workers = cpu_count()
        mp_pool = Pool(num_workers)

        # logging variables
        best_test_ratio = 0
        running_reward = 0
        critic_loss = []
        avg_length = 0
        timestep = 0
        prev_time = time.time()

        # training loop
        for i_episode in range(1, args.max_episodes + 1):
            items_batch = ItemsContainer()
            for b in range(args.batch_size):
                graph_index = ((i_episode - 1) * args.batch_size + b) % len(tuples_train)
                inp_graph, ori_greedy, _, baselines = tuples_train[graph_index]  # we treat inp_graph as the state
                greedy = ori_greedy
                forward_edge_candidates, backward_edge_candidates = dag_graph.get_edge_candidates(inp_graph, init=True)
                items_batch.append(0, inp_graph, greedy, forward_edge_candidates, backward_edge_candidates, False, ori_greedy)

            for t in range(args.max_timesteps):
                timestep += 1

                # Running policy_old:
                with torch.no_grad():
                    action_batch = ppo.policy_old.act(items_batch.inp_graph, items_batch.forward_edge_candidates, items_batch.backward_edge_candidates, memory)

                def step_func_feeder(batch_size):
                    batch_inp_graph = items_batch.inp_graph
                    action_batch_cpu = action_batch.cpu()
                    batch_greedy = items_batch.greedy
                    for b in range(batch_size):
                        yield batch_inp_graph[b], action_batch_cpu[:, b], batch_greedy[b]

                if args.batch_size > 1:
                    pool_map = mp_pool.starmap_async(dag_graph.step, step_func_feeder(args.batch_size))
                    step_list = pool_map.get()
                else:
                    step_list = [dag_graph.step(*x) for x in step_func_feeder(args.batch_size)]
                for b, item in enumerate(step_list):
                    reward, inp_graph, greedy, forward_edge_candidates, backward_edge_candidates, done = item
                    if t == args.max_timesteps - 1:
                        done = True
                    items_batch.update(b, reward=reward, inp_graph=inp_graph, greedy=greedy,
                                          forward_edge_candidates=forward_edge_candidates, 
                                          backward_edge_candidates=backward_edge_candidates, done=done)

                # Saving reward and is_terminal:
                memory.rewards.append(items_batch.reward)
                memory.is_terminals.append(items_batch.done)
                
                # update if its time
                if timestep % args.update_timestep == 0:
                    
                    closs = ppo.update(memory)
                    critic_loss.append(closs)
                    if summary_writer:
                        # summary_writer.add_scalar('critic mse/train', closs, timestep)
                        with summary_writer.as_default():  # TensorFlow에서 summary 작성은 이 블록 안에서 수행
                            if closs.is_cuda:
                                closs_value = closs.cpu().item()
                            else:
                                closs_value = closs.item()
                            tf.summary.scalar('critic mse/train', closs_value, step=timestep)
                    memory.clear_memory()

                running_reward += sum(items_batch.reward) / args.batch_size
                if any(items_batch.done):
                    break

            avg_length += t+1

            # logging
            if i_episode % args.log_interval == 0:
                avg_length = avg_length / args.log_interval
                running_reward = running_reward / args.log_interval
                if len(critic_loss) > 0:
                    critic_loss = torch.mean(torch.stack(critic_loss))
                else:
                    critic_loss = -1
                now_time = time.time()
                avg_time = (now_time - prev_time) / args.log_interval
                prev_time = now_time

                if summary_writer:
                    # summary_writer.add_scalar('reward/train', running_reward, timestep)
                    tf.summary.scalar('reward/train', running_reward, step=timestep)
                    # summary_writer.add_scalar('time/train', avg_time, timestep)
                    tf.summary.scalar('reward/train', avg_time, step=timestep)
                    
                    for lr_id, x in enumerate(ppo.optimizer.param_groups):
                        # summary_writer.add_scalar(f'lr/{lr_id}', x['lr'], timestep)
                        tf.summary.scalar('lr/{lr_id}', x['lr'], step=timestep)

                print(
                    f'Episode {i_episode} \t '
                    f'avg length: {avg_length:.2f} \t '
                    f'critic mse: {critic_loss:.4f} \t '
                    f'reward: {running_reward:.4f} \t '
                    f'time per episode: {avg_time:.2f}'
                )

                #######
                
                cpu_usage = psutil.cpu_percent(interval=1)

                # 현재 메모리 사용량 (전체 시스템 기준)
                virt_memory = psutil.virtual_memory()
                total_memory = virt_memory.total / (1024 ** 3)  # 전체 메모리 (MB)
                used_memory = virt_memory.used / (1024 ** 3)    # 사용된 메모리 (MB)
                free_memory = virt_memory.available / (1024 ** 3)  # 사용 가능한 메모리 (MB)

                # 출력
                print(f"CPU Usage: {cpu_usage}%")
                print(f"Total Memory: {total_memory:.2f} GB")
                print(f"Used Memory: {used_memory:.2f} GB")
                ######

                running_reward = 0
                avg_length = 0
                critic_loss = []

            # testing
            if i_episode % args.test_interval == 0:
                with torch.no_grad():
                    # record time spent on test
                    prev_test_time = time.time()
                    #print("########## Evaluate on Train ##########")
                    #train_dict = evaluate(ppo.policy, dag_graph, tuples_train, args.max_timesteps, args.search_size, mp_pool)
                    #for key, val in train_dict.items():
                    #    if isinstance(val, dict):
                    #        if summary_writer:
                    #            summary_writer.add_scalars(f'{key}/train-eval', val, timestep)
                    #    else:
                    #        if summary_writer:
                    #            summary_writer.add_scalar(f'{key}/train-eval', val, timestep)
                    print("########## Evaluate on Test ##########")
                    # run testing
                    test_dict = evaluate(ppo.policy, dag_graph, tuples_test, args.max_timesteps, args.search_size, mp_pool)
                    # write to summary writter
                    # for key, val in test_dict.items():
                    #     if isinstance(val, dict):
                    #         if summary_writer:
                    #             # summary_writer.add_scalars(f'{key}/test', val, timestep)
                    #             tf.summary.scalar('{key}/test', float(val), step=timestep)
                    #     else:
                    #         if summary_writer:
                    #             # summary_writer.add_scalar(f'{key}/test', val, timestep)
                    #             tf.summary.scalar('{key}/test',float(val), step=timestep)
                    print("########## Evaluate complete ##########")
                    # fix running time value
                    prev_time += time.time() - prev_test_time

                if test_dict["ratio"]["mean"] > best_test_ratio:
                    best_test_ratio = test_dict["ratio"]["mean"]
                    if not os.path.exists('./results'):
                        os.makedirs('./results', exist_ok=True)
                    file_name = f'./results/PPO_{args.scheduler_type}_dag_num{args.num_init_dags}' \
                                f'_beam{args.search_size}_ratio{best_test_ratio:.4f}.pt'
                    torch.save(ppo.policy.state_dict(), file_name)
    elif args.model == "gfn":
        # FL-DB Implementation
        gfn = GFN(dag_graph, args, device)
        num_workers = cpu_count()
        mp_pool = Pool(num_workers)
        
        # logging variables
        best_test_ratio = 0
        running_reward = 0
        gfn_loss = []
        avg_length = 0
        timestep = 0
        prev_time = time.time()
        
        # training loop
        for i_episode in range(1, args.max_episodes + 1):

            items_batch = ItemsContainer()
            for b in range(args.batch_size):
                graph_index = ((i_episode - 1) * args.batch_size + b) % len(tuples_train)
                inp_graph, ori_greedy, _, baselines = tuples_train[graph_index]  # we treat inp_graph as the state
                greedy = ori_greedy
                forward_edge_candidates, backward_edge_candidates = dag_graph.get_edge_candidates(inp_graph, init=True)
                items_batch.append(0, inp_graph, greedy, forward_edge_candidates, backward_edge_candidates, False, ori_greedy)
            
            for t in range(args.max_timesteps):
                timestep += 1
                
                with torch.no_grad():
                    action_batch = gfn.act(items_batch.inp_graph, items_batch.forward_edge_candidates, items_batch.backward_edge_candidates, memory)
                    # if graph_index not in list(memory.keys()):
                    #     memory[graph_index] = Memory_deque(maxlen=100)
                    # action_batch = gfn.act(items_batch.inp_graph, items_batch.edge_candidates, memory[graph_index])
                    
                def step_func_feeder(batch_size):
                    batch_inp_graph = items_batch.inp_graph
                    action_batch_cpu = action_batch.cpu()
                    batch_greedy = items_batch.greedy
                    for b in range(batch_size):
                        yield batch_inp_graph[b], action_batch_cpu[:, b], batch_greedy[b]
                        
                if args.batch_size > 1:
                    pool_map = mp_pool.starmap_async(dag_graph.step, step_func_feeder(args.batch_size))
                    step_list = pool_map.get()
                else:
                    step_list = [dag_graph.step(*x) for x in step_func_feeder(args.batch_size)]
                for b, item in enumerate(step_list):
                    reward, inp_graph, greedy, forward_edge_candidates, backward_edge_candidates, done = item
                    if t == args.max_timesteps - 1:
                        done = True
                    # items_batch.update(b, reward=reward, inp_graph=inp_graph, greedy=greedy,
                    #                 edge_candidates=edge_candidates, done=done)
                    items_batch.update(b, reward=reward, inp_graph=inp_graph, greedy=greedy,
                                    forward_edge_candidates=forward_edge_candidates, backward_edge_candidates=backward_edge_candidates, done=done)
                
                # Saving reward and is_terminal:
                memory.rewards.append(items_batch.reward)
                memory.is_terminals.append(items_batch.done)
                memory.next_states.append(items_batch.inp_graph)  
                # memory[graph_index].rewards.append(items_batch.reward)
                # memory[graph_index].is_terminals.append(items_batch.done)
                # memory[graph_index].next_states.append(items_batch.inp_graph)     

                # update if its time
                if timestep % args.update_timestep == 0:
                    # sampling instance and then sampling data 
                    # instance_idx = random.randint(0, len(memory)-1)
                    # sampled_memory = sample_memory(memory[instance_idx])
                    # if random.random() < 0.5:
                    sampled_memory = sample_memory(memory)
                    loss = gfn.update(sampled_memory)
                    gfn_loss.append(loss)

                    if summary_writer:
                        with summary_writer.as_default():
                            if loss.is_cuda:
                                loss_value = loss.cpu().item()
                            else:
                                loss_value = loss.item()
                            tf.summary.scalar('gfn loss/train', loss_value, step=timestep)
                        
                    # sample filtering
                    # else:
                    #     sampled_memory = sample_recent_memory(memory,20)
                    #     memory.trim_recent(20)
                    #     loss = gfn.update(sampled_memory)
                    #     gfn_loss.append(loss)
                    #     if summary_writer:
                    #         with summary_writer.as_default():
                    #             if loss.is_cuda:
                    #                 loss_value = loss.cpu().item()
                    #             else:
                    #                 loss_value = loss.item()
                    #             tf.summary.scalar('gfn loss/train', loss_value, step=timestep)
                    #     filtered_sampled_memory = filter_memory(sampled_memory)
                    #     memory.merge(filtered_sampled_memory)


                running_reward += sum(items_batch.reward) / args.batch_size
                if any(items_batch.done):
                    break
            
            avg_length += t+1
            
            # logging
            if i_episode % args.log_interval == 0:
                avg_length = avg_length / args.log_interval
                running_reward = running_reward / args.log_interval
                if len(gfn_loss) > 0:
                    gfn_loss = torch.mean(torch.stack(gfn_loss))
                else:
                    gfn_loss = -1
                now_time = time.time()
                avg_time = (now_time - prev_time) / args.log_interval
                prev_time = now_time

                if summary_writer:
                    # summary_writer.add_scalar('reward/train', running_reward, timestep)
                    tf.summary.scalar('reward/train', running_reward, step=timestep)
                    # summary_writer.add_scalar('time/train', avg_time, timestep)
                    tf.summary.scalar('reward/train', avg_time, step=timestep)
                    
                    for lr_id, x in enumerate(gfn.optimizer.param_groups):
                        # summary_writer.add_scalar(f'lr/{lr_id}', x['lr'], timestep)
                        tf.summary.scalar('lr/{lr_id}', x['lr'], step=timestep)

                print(
                    f'Episode {i_episode} \t '
                    f'avg length: {avg_length:.2f} \t '
                    f'gfn_loss: {gfn_loss:.4f} \t '
                    f'reward: {running_reward:.4f} \t '
                    f'time per episode: {avg_time:.2f}'
                )
                
                running_reward = 0
                avg_length = 0
                gfn_loss = []
                
                #######
                
                cpu_usage = psutil.cpu_percent(interval=1)

                # 현재 메모리 사용량 (전체 시스템 기준)
                virt_memory = psutil.virtual_memory()
                total_memory = virt_memory.total / (1024 ** 3)  # 전체 메모리 (MB)
                used_memory = virt_memory.used / (1024 ** 3)    # 사용된 메모리 (MB)
                free_memory = virt_memory.available / (1024 ** 3)  # 사용 가능한 메모리 (MB)

                # 출력
                print(f"CPU Usage: {cpu_usage}%  Total Memory: {total_memory:.2f} GB  Used Memory: {used_memory:.2f} GB ")
        
            # testing
            if i_episode % args.test_interval == 0:
                with torch.no_grad():
                    # record time spent on test
                    prev_test_time = time.time()
                    #print("########## Evaluate on Train ##########")
                    #train_dict = evaluate(ppo.policy, dag_graph, tuples_train, args.max_timesteps, args.search_size, mp_pool)
                    #for key, val in train_dict.items():
                    #    if isinstance(val, dict):
                    #        if summary_writer:
                    #            summary_writer.add_scalars(f'{key}/train-eval', val, timestep)
                    #    else:
                    #        if summary_writer:
                    #            summary_writer.add_scalar(f'{key}/train-eval', val, timestep)
                    print("########## Evaluate on Test ##########")
                    # run testing
                    test_dict = evaluate_gfn(gfn, dag_graph, tuples_test, args.max_timesteps, args.search_size, mp_pool)
                    # write to summary writter
                    # for key, val in test_dict.items():
                    #     if isinstance(val, dict):
                    #         if summary_writer:
                    #             # summary_writer.add_scalars(f'{key}/test', val, timestep)
                    #             tf.summary.scalar('{key}/test', float(val), step=timestep)
                    #     else:
                    #         if summary_writer:
                    #             # summary_writer.add_scalar(f'{key}/test', val, timestep)
                    #             tf.summary.scalar('{key}/test',float(val), step=timestep)
                    print("########## Evaluate complete ##########")
                    # fix running time value
                    prev_time += time.time() - prev_test_time

                if test_dict["ratio"]["mean"] > best_test_ratio:
                    best_test_ratio = test_dict["ratio"]["mean"]
                    # file_name = f'./GFN_{args.scheduler_type}_dag_num{args.num_init_dags}' \
                    #             f'_beam{args.search_size}_ratio{best_test_ratio:.4f}.pt'
                    if not os.path.exists('./results'):
                        os.makedirs('./results', exist_ok=True)
                    file_name = f'./results/GFN_{args.scheduler_type}_dag_num{args.num_init_dags}' \
                                f'_beam{args.search_size}_ratio{best_test_ratio:.4f}.pt'
                    torch.save(gfn.state_dict(), file_name)
                    

def parse_arguments():
    parser = argparse.ArgumentParser(description='DAG scheduler. You have two ways of setting the parameters: \n'
                                                 '1) set parameters by command line arguments \n'
                                                 '2) specify --config path/to/config.yaml')
    # environment configs
    parser.add_argument('--scheduler_type', default='sft')
    parser.add_argument('--resource_limit', default=600, type=float)
    parser.add_argument('--add_graph_features', action='store_true')
    parser.add_argument('--num_init_dags', default=5, type=int)
    # parser.add_argument('--max_selected_edges', default=3, type=int)
    parser.add_argument('--gamma', default=0.99, type=float, help='discount factor for accumulated reward')
    parser.add_argument('--train_sample', default=50, type=int, help='number of training samples')
    parser.add_argument('--test_sample', default=50, type=int, help='number of testing samples')

    # decode(testing) configs
    # parser.add_argument('--beam_search', action='store_true')
    parser.add_argument('--search_size', default=5, type=int)

    # learning configs
    parser.add_argument('--learning_rate', default=0.002, type=float)
    parser.add_argument('--lr_steps', default=[], type=list)
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size when sampling')
    parser.add_argument('--betas', default=(0.9, 0.999), help='Adam optimizer\'s beta')
    parser.add_argument('--max_episodes', default=50000, type=int, help='max training episodes')
    parser.add_argument('--max_timesteps', default=300, type=int, help='max timesteps in one episode')
    parser.add_argument('--update_timestep', default=2000, type=int, help='update policy every n timesteps')
    parser.add_argument('--k_epochs', default=4, type=int, help='update policy for K epochs')
    parser.add_argument('--eps_clip', default=0.2, type=float, help='clip parameter for PPO')

    # model parameters
    parser.add_argument('--one_hot_degree', default=0, type=int)
    parser.add_argument('--batch_norm', action='store_true')
    parser.add_argument('--node_output_size', default=16, type=int)
    parser.add_argument('--gnn_layers', default=10, type=int, help='number of GNN layers')

    # misc configs
    parser.add_argument('--config', default=None, type=str, help='path to config file,'
                        ' and command line arguments will be overwritten by the config file')
    parser.add_argument('--random_seed', default=None, type=int)
    parser.add_argument('--test_interval', default=500, type=int, help='run testing in the interval (episodes)')
    parser.add_argument('--log_interval', default=100, type=int, help='print avg reward in the interval (episodes)')
    parser.add_argument('--test_model_weight', default='', type=str, help='the path of model weight to be loaded')
    
    
    # GFlowNet
    parser.add_argument('--model', default='ppo', type=str, help='model name')

    args = parser.parse_args()

    if args.config:
        with open('config/' + args.config) as f:
            cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
            for key, val in cfg_dict.items():
                assert hasattr(args, key), f'Unknown config key: {key}'
                setattr(args, key, val)
            f.seek(0)
            print(f'Config file: {args.config}', )
            for line in f.readlines():
                print(line.rstrip())

    print_args(args)

    return args
if __name__ == '__main__':
    main(parse_arguments()) 
# if __name__ == '__main__':
    
#     log_file_path = "dag_gfn_100_priori-Sampling_beta10.txt"

#     # stdout을 텍스트 파일로 리디렉션
#     with open(log_file_path, "w") as log_file:
#         sys.stdout = log_file  # stdout을 log_file로 설정

#         #메인 함수 실행
#         start_time = time.time()
#         main(parse_arguments())
            
#         # 실행 시간 기록
#         print(f'total elapsed time: {time.time() - start_time}')

#     # stdout을 원래대로 복구
#     sys.stdout = sys.__stdout__

