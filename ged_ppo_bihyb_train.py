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
import wandb
from src.ged_ppo_bihyb_model import ActorNet, CriticNet, GraphEncoder
from utils.utils import print_args
from utils.tfboard_helper import TensorboardUtil
from ged_ppo_bihyb_eval import evaluate,evaluate_gfn
from utils.ged_env import GEDenv
import psutil
from collections import deque
wandb.login(key="89cf62dcabfd331d496f3c5f278a9388394a25d4")
wandb.init(project='GED_PPO_GFN')
wandb.run.name = 'GFN_on-policy_AIDS20-30'

class ItemsContainer:
    def __init__(self):
        self.__reward = []
        self.__inp_graph_1 = []
        self.__inp_graph_2 = []
        self.__ori_k = []
        self.__ori_greedy = []
        self.__forward_edge_candidates = []
        self.__backward_edge_candidates = []
        self.__greedy = []
        self.__done = []

    def append(self, reward, inp_graph_1, inp_graph_2, ori_k, greedy,forward_edge_candidates, backward_edge_candidates, done, ori_greedy):
        self.__reward.append(reward)
        self.__inp_graph_1.append(inp_graph_1)
        self.__inp_graph_2.append(inp_graph_2)
        self.__ori_k.append(ori_k)
        self.__forward_edge_candidates.append(forward_edge_candidates)
        self.__backward_edge_candidates.append(backward_edge_candidates)
        self.__greedy.append(greedy)
        self.__done.append(done)
        self.__ori_greedy.append(ori_greedy)

    @property
    def reward(self):
        return deepcopy(self.__reward)

    @property
    def inp_graph_1(self):
        return deepcopy(self.__inp_graph_1)

    @property
    def inp_graph_2(self):
        return deepcopy(self.__inp_graph_2)

    @property
    def ori_k(self):
        return deepcopy(self.__ori_k)

    @property
    def forward_edge_candidates(self):
        return deepcopy(self.__forward_edge_candidates)
    
    @property
    def backward_edge_candidates(self):
        return deepcopy(self.__backward_edge_candidates)
        
    @property
    def greedy(self):
        return deepcopy(self.__greedy)

    @property
    def done(self):
        return deepcopy(self.__done)

    @property
    def ori_greedy(self):
        return deepcopy(self.__ori_greedy)

    def update(self, idx, reward=None, inp_graph_1=None, inp_graph_2=None, greedy=None, forward_edge_candidates=None, backward_edge_candidates=None, done=None, ori_greedy=None):
        if reward is not None:
            self.__reward[idx] = reward
        if inp_graph_1 is not None:
            self.__inp_graph_1[idx] = inp_graph_1
        if inp_graph_2 is not None:
            self.__inp_graph_2[idx] = inp_graph_2
        if greedy is not None:
            self.__greedy[idx] = greedy
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
        # self.candidates = [] 
        self.forward_candidates = []
        self.backward_candidates = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        # del self.candidates[:]
        del self.forward_candidates[:]
        del self.backward_candidates[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class Memory_deque:
    def __init__(self, maxlen=300):  # maxlen 기본값 설정
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

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # 안정성을 위해 최대값을 빼줌 (overflow 방지)
    return exp_x / exp_x.sum()


def sample_memory(memory, sample_size=10):
    
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
        sampled_memory.candidates.append(memory.candidates[idx])
        #sampled_memory.forward_candidates.append(memory.forward_candidates[idx])
        #sampled_memory.backward_candidates.append(memory.backward_candidates[idx])
        sampled_memory.logprobs.append(memory.logprobs[idx])
        sampled_memory.rewards.append(memory.rewards[idx])
        sampled_memory.is_terminals.append(memory.is_terminals[idx])

    sampled_rewards = np.array(sampled_memory.rewards, dtype=np.float32)
    # print(sampled_rewards)
    return sampled_memory


class ActorCritic(nn.Module):
    def __init__(self, node_feature_dim, node_output_size, batch_norm, one_hot_degree, gnn_layers):
        super(ActorCritic, self).__init__()

        self.state_encoder = GraphEncoder(node_feature_dim, node_output_size, batch_norm, one_hot_degree, gnn_layers)
        self.actor_net = ActorNet(node_output_size, batch_norm)
        self.value_net = CriticNet(node_output_size, batch_norm)

    def forward(self):
        raise NotImplementedError

    def act(self, inp_graph_1, inp_graph_2,forward_edge_candidates, backward_edge_candidates, memory):
        diff_feat, graph_feat_1, graph_feat_2 = self.state_encoder(inp_graph_1, inp_graph_2)
        actions, action_logits, entropy = self.actor_net(diff_feat)

        memory.states.append((inp_graph_1, inp_graph_2))
        memory.actions.append(actions)
        memory.logprobs.append(action_logits)

        memory.forward_candidates.append(forward_edge_candidates)
        memory.backward_candidates.append(backward_edge_candidates)

        return actions

    def evaluate(self, inp_graph_1, inp_graph_2, action):
        diff_feat, graph_feat_1, graph_feat_2 = self.state_encoder(inp_graph_1, inp_graph_2)
        _, action_logits, entropy = self.actor_net(diff_feat, action)
        state_value = self.value_net(graph_feat_1, graph_feat_2)
        return action_logits, state_value, entropy


class PPO:
    def __init__(self, args, device):
        self.lr = args.learning_rate
        self.betas = args.betas
        self.gamma = args.gamma
        self.eps_clip = args.eps_clip
        self.K_epochs = args.k_epochs

        self.device = device

        ac_params = args.node_feature_dim, args.node_output_size, args.batch_norm, args.one_hot_degree, args.gnn_layers

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
                self.policy.evaluate(*memory.states[-1], memory.actions[-1].to(self.device))
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
        old_graph_1 = []
        old_graph_2 = []
        for state in memory.states:
            old_graph_1 += state[0]
            old_graph_2 += state[1]
        old_actions = torch.cat(memory.actions, dim=1)
        old_logprobs = torch.cat(memory.logprobs, dim=1)
        #old_candidates = []
        #for candi in memory.candidates:
        #    old_candidates += candi

        critic_loss_sum = 0

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_graph_1, old_graph_2, old_actions)
            
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



class GFN:
    def __init__(self, args, device):
        self.lr = args.learning_rate
        self.betas = args.betas
        self.device = device

        ac_params = args.node_feature_dim, args.node_output_size, args.batch_norm, args.one_hot_degree, args.gnn_layers

        self.state_encoder = GraphEncoder(args.node_feature_dim, args.node_output_size, args.batch_norm, args.one_hot_degree, args.gnn_layers)
        self.forward_policy = ActorNet(args.node_output_size, args.batch_norm)
        self.flow_model = CriticNet(args.node_output_size, args.batch_norm)

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

    
    def act(self, inp_graph_1, inp_graph_2, forward_edge_candidates, backward_edge_candidates, memory):
        diff_feat, graph_feat_1, graph_feat_2 = self.state_encoder(inp_graph_1, inp_graph_2)
        actions, action_logits, entropy = self.forward_policy(diff_feat)

        memory.states.append((inp_graph_1, inp_graph_2))
        memory.actions.append(actions)
        memory.logprobs.append(action_logits)
        memory.forward_candidates.append(forward_edge_candidates)
        memory.backward_candidates.append(backward_edge_candidates)
        
        return actions
    
    
    def evaluate(self, inp_graph_1, inp_graph_2, action):
        diff_feat, graph_feat_1, graph_feat_2 = self.state_encoder(inp_graph_1, inp_graph_2)
        _, action_logits, entropy = self.forward_policy(diff_feat, action)
        state_value = self.flow_model(graph_feat_1, graph_feat_2)
        return action_logits# , state_value, entropy


    def update(self, memory):
        states = []
        for state in memory.states:
            states.append(state) 
        

        actions = torch.cat(list(memory.actions), dim=1)
        rewards = torch.tensor(list(memory.rewards), dtype=torch.float32).to(self.device).flatten()
        # next_states = []
        # for state in list(memory.next_states):
        #     next_states += state
        forward_candidates = []
        for candi in list(memory.forward_candidates):
            forward_candidates += candi
        backward_candidates = []
        for candi in list(memory.backward_candidates):
            backward_candidates += candi

        # log p(a1, a2) = log p(a1) + log p(a2|a1)
        states1, states2 = zip(*states)
        
        states1 = [data for sublist in list(states1) for data in sublist]
        states2 = [data for sublist in list(states2) for data in sublist]
 
        log_pf = self.evaluate(list(states1), list(states2), actions).sum(dim=0)
        # we set pb as uniform policy. pb = 1/(number of possible backward actions from current state)
        log_pb = torch.tensor([1/(sum([len(v) for k, v in candi.items()]) // 2 + 1e-7) for candi in backward_candidates]).to(self.device).log()
        _, state_feats1, state_feats2 = self.state_encoder(states1,states2)
        log_fs = self.flow_model(state_feats1, state_feats2)

        total_loss = torch.zeros((len(states)-1)).to(self.device)
        total_loss += log_fs[:-1]
        total_loss += log_pf[:-1]
        total_loss -= log_pb[1:]
        total_loss -= log_fs[1:]
        
        # FL-DB parametrization. We consider reward as intermediate energy function over transitions
        total_loss -= rewards[:-1]
        total_loss = total_loss.pow(2).mean()
        
        # back propagation
        self.optimizer.zero_grad()
        total_loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.state_encoder.parameters(), 5)
        torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), 5)
        torch.nn.utils.clip_grad_norm_(self.forward_policy.parameters(), 5)
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

    # create environment
    ged_env = GEDenv(args.solver_type, args.dataset)
    args.node_feature_dim = ged_env.feature_dim
    # get current device (cuda or cpu)
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

    # load training/testing data
    tuples_train = ged_env.generate_tuples(ged_env.training_graphs, args.train_sample, 0, device)
    tuples_test = ged_env.generate_tuples(ged_env.val_graphs, args.test_sample, 1, device)

    # init models
    # memory = Memory()
    memory =  Memory_deque(maxlen=300) 
    if args.model == 'ppo':
        ppo = PPO(args, device)
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
                inp_graph_1, inp_graph_2, ori_k, ori_greedy, baselines, _ = tuples_train[graph_index]
                greedy = ori_greedy
                forward_edge_candidates, backward_edge_candidates = ged_env.get_edge_candidates(inp_graph_1, init=True)
                items_batch.append(0, inp_graph_1, inp_graph_2, ori_k, greedy,forward_edge_candidates, backward_edge_candidates, False, ori_greedy)

            for t in range(args.max_timesteps):
                timestep += 1

                # Running policy_old:
                with torch.no_grad():
                    action_batch = ppo.policy_old.act(items_batch.inp_graph_1, items_batch.inp_graph_2, items_batch.forward_edge_candidates, items_batch.backward_edge_candidates, memory)

                def step_func_feeder(batch_size):
                    batch_inp_graph_1 = items_batch.inp_graph_1
                    batch_inp_graph_2 = items_batch.inp_graph_2
                    batch_ori_k = items_batch.ori_k
                    batch_greedy = items_batch.greedy
                    for b in range(batch_size):
                        yield batch_inp_graph_1[b], batch_inp_graph_2[b], batch_ori_k[b], action_batch[:, b], batch_greedy[b], args.beta

                if args.batch_size > 1:
                    pool_map = mp_pool.starmap_async(ged_env.step, step_func_feeder(args.batch_size))
                    step_list = pool_map.get()
                else:
                    step_list = [ged_env.step(*x) for x in step_func_feeder(args.batch_size)]
                for b, item in enumerate(step_list):
                    reward, inp_graph_1, greedy, forward_edge_candidates, backward_edge_candidates = item  # only graph1 is updated
                    if t == args.max_timesteps - 1:
                        done = True
                    else:
                        done = False
                    items_batch.update(b, reward=reward, inp_graph_1=inp_graph_1, greedy=greedy, forward_edge_candidates = forward_edge_candidates, backward_edge_candidates = backward_edge_candidates, done=done)

                # Saving reward and is_terminal:
                memory.rewards.append(items_batch.reward)
                memory.is_terminals.append(items_batch.done)

                # update if its time
                if timestep % args.update_timestep == 0:
                    closs = ppo.update(memory)
                    critic_loss.append(closs)
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

          
                print(
                    f'Episode {i_episode} \t '
                    f'avg length: {avg_length:.2f} \t '
                    f'critic mse: {critic_loss:.4f} \t '
                    f'reward: {running_reward.item():.4f} \t '
                    f'time per episode: {avg_time:.2f}'
                )
                #############################
                
                cpu_usage = psutil.cpu_percent(interval=1)

                # 현재 메모리 사용량 (전체 시스템 기준)
                virt_memory = psutil.virtual_memory()
                total_memory = virt_memory.total / (1024 ** 3)  # 전체 메모리 (MB)
                used_memory = virt_memory.used / (1024 ** 3)    # 사용된 메모리 (MB)
                free_memory = virt_memory.available / (1024 ** 3)  # 사용 가능한 메모리 (MB)

                # 출력
                print(f"CPU Usage: {cpu_usage}% Total Memory: {total_memory:.2f} GB  Used Memory: {used_memory:.2f} GB")
                ######
                wandb.log({
                    "Episode": i_episode,
                    "avg_length": avg_length,
                    "critic mse": critic_loss,
                    "reward": running_reward,
                    "time_per_episode": avg_time,
                    "CPU Usage": cpu_usage,
                    "Total Memory": total_memory,
                    "Used Memory": used_memory,
                },step=i_episode)
                #############################
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
                    test_dict = evaluate(ppo.policy, ged_env, tuples_test, args.max_timesteps, args.search_size,
                                        None if torch.cuda.is_available() else mp_pool)
                    # write to summary writter
                    #for key, val in test_dict.items():
                    #    if isinstance(val, dict):
                    #        if summary_writer:
                    #            summary_writer.add_scalars(f'{key}/test', val, timestep)
                    #    else:
                    #        if summary_writer:
                    #            summary_writer.add_scalar(f'{key}/test', val, timestep)
                    print("########## Evaluate complete ##########")
                    # fix running time value
                    prev_time += time.time() - prev_test_time
                wandb.log({
                        "mean_ratio": test_dict["ratio"]["mean"]})
                if test_dict["ratio"]["mean"] > best_test_ratio:
                    best_test_ratio = test_dict["ratio"]["mean"]
                    file_name = f'./PPO_{args.solver_type}_dataset{args.dataset}' \
                                f'_beam{args.search_size}_ratio{best_test_ratio:.4f}.pt'
                    torch.save(ppo.policy.state_dict(), file_name)
    elif args.model == 'gfn':
        gfn = GFN(args, device)
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
                inp_graph_1, inp_graph_2, ori_k, ori_greedy, baselines, _ = tuples_train[graph_index]
                greedy = ori_greedy

                forward_edge_candidates, backward_edge_candidates = ged_env.get_edge_candidates(inp_graph_1, init=True)
                items_batch.append(0, inp_graph_1, inp_graph_2, ori_k, greedy, forward_edge_candidates, backward_edge_candidates, False, ori_greedy)

            for t in range(args.max_timesteps):
                timestep += 1

                # Running policy_old:
                
                with torch.no_grad():
                    action_batch = gfn.act(items_batch.inp_graph_1, items_batch.inp_graph_2, items_batch.forward_edge_candidates, items_batch.backward_edge_candidates, memory)
                
                def step_func_feeder(batch_size):
                    batch_inp_graph_1 = items_batch.inp_graph_1
                    batch_inp_graph_2 = items_batch.inp_graph_2
                    batch_ori_k = items_batch.ori_k
                    batch_greedy = items_batch.greedy
                    for b in range(batch_size):
                        yield batch_inp_graph_1[b], batch_inp_graph_2[b], batch_ori_k[b], action_batch[:, b], batch_greedy[b], args.beta

                if args.batch_size > 1:
                    pool_map = mp_pool.starmap_async(ged_env.step, step_func_feeder(args.batch_size))
                    step_list = pool_map.get()
                else:
                    step_list = [ged_env.step(*x) for x in step_func_feeder(args.batch_size)]
                for b, item in enumerate(step_list):
                    reward, inp_graph_1, greedy,forward_edge_candidates, backward_edge_candidates = item  # only graph1 is updated
                    if t == args.max_timesteps - 1:
                        done = True
                    else:
                        done = False
                    items_batch.update(b, reward=reward, inp_graph_1=inp_graph_1, greedy=greedy,
                                    forward_edge_candidates=forward_edge_candidates, backward_edge_candidates=backward_edge_candidates, done=done)
                    
                # Saving reward and is_terminal:
                memory.rewards.append(items_batch.reward)
                memory.is_terminals.append(items_batch.done)
                # memory.next_states.append((items_batch.inp_graph_1, items_batch.inp_graph_2))

                # update if its time
                if timestep % args.update_timestep == 0:
                    sampled_memory = sample_memory(memory)
                    loss = gfn.update(sampled_memory)
                    # loss = gfn.update(memory)
                    gfn_loss.append(loss)                      

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


                print(
                    f'Episode {i_episode} \t '
                    f'avg length: {avg_length:.2f} \t '
                    f'gfn mse: {gfn_loss:.4f} \t '
                    f'reward: {running_reward.item():.4f} \t '
                    f'time per episode: {avg_time:.2f}'
                )
                
                # logging memory and cpu                
                cpu_usage = psutil.cpu_percent(interval=1)
                virt_memory = psutil.virtual_memory()
                print(f"CPU Usage: {cpu_usage}% Total Memory: {virt_memory.total / (1024 ** 3):.2f} GB  Used Memory: {virt_memory.used / (1024 ** 3) :.2f} GB")
                if virt_memory.used / (1024 ** 3) >=200:
                    break

                # logging on wandb
                wandb.log({
                    "Episode": i_episode,
                    "avg_length": avg_length,
                    "gfn_loss": gfn_loss,
                    "reward": running_reward,
                    "time_per_episode": avg_time,
                    "CPU Usage": cpu_usage,
                    "Used Memory": virt_memory.used / (1024 ** 3),
                },step=i_episode)
                
                running_reward = 0
                avg_length = 0
                gfn_loss = []

            # testing
            if i_episode % args.test_interval == 0:
                with torch.no_grad():
                    # record time spent on test
                    prev_test_time = time.time()
                    print("########## Evaluate on Test ##########")
                    
                    # run testing
                    test_dict = evaluate_gfn(gfn, ged_env, tuples_test, args.max_timesteps, args.search_size,
                                        None if torch.cuda.is_available() else mp_pool)
                    print("########## Evaluate complete ##########")
                    # fix running time value
                    prev_time += time.time() - prev_test_time

                wandb.log({
                        "mean_ratio": test_dict["ratio"]["mean"]})
                if test_dict["ratio"]["mean"] > best_test_ratio:
                    best_test_ratio = test_dict["ratio"]["mean"]
                    file_name = f'./GFN_{args.solver_type}_dataset{args.dataset}' \
                                f'_beam{args.search_size}_ratio{best_test_ratio:.4f}.pt'
                    torch.save(gfn.state_dict(), file_name)
    
def parse_arguments():
    parser = argparse.ArgumentParser(description='GED solver. You have two ways of setting the parameters: \n'
                                                 '1) set parameters by command line arguments \n'
                                                 '2) specify --config path/to/config.yaml')
    # environment configs
    parser.add_argument('--solver_type', default='hungarian')
    parser.add_argument('--resource_limit', default=600, type=float)
    parser.add_argument('--add_graph_features', action='store_true')
    parser.add_argument('--dataset', default='AIDS700nef')
    parser.add_argument('--gamma', default=0.99, type=float, help='discount factor for accumulated reward')
    parser.add_argument('--train_sample', default=50, type=int, help='number of training samples')
    parser.add_argument('--test_sample', default=10, type=int, help='number of testing samples')

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
    parser.add_argument('--update_timestep', default=20, type=int, help='update policy every n timesteps')
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
    parser.add_argument('--model', default='gfn', type=str, help='model name')
    parser.add_argument('--beta', default=1, type=int, help='gfn hyper parameter')

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
