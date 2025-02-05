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