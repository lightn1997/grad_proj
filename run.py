"""Play Othello."""
import argparse
from posix import environ
import othello
import policies
from collections import deque
import numpy as np
import pandas as pd
import openpyxl


def create_policy(policy_type='rand', board_size=8, seed=0, search_depth=1, learned=False, q_values = None, epi = 0):
    if policy_type == 'rand':
        policy = policies.RandomPolicy(seed=seed)
    elif policy_type == 'greedy':
        policy = policies.GreedyPolicy()
    elif policy_type == 'minimax':
        policy = policies.MaxiMinPolicy(search_depth)
    elif policy_type == 'DQN':
        policy = policies.DQN(board_size=board_size, learned=learned, q_values=q_values, epi = epi)
    else:
        policy = policies.HumanPolicy(board_size)
    return policy


def play(protagonist,
         protagonist_agent_type='greedy',
         opponent_agent_type='rand',
         board_size=8,
         num_rounds=100,
         protagonist_search_depth=1,
         opponent_search_depth=1,
         protagonist_learned=False,
         protagonist_Qvalues=None,
         protagonist_learned_games=0,
         opponent_learned=False,
         opponent_Qvalues=None,
         opponent_learned_games=0,
         rand_seed=0,
         env_init_rand_steps=0,
         num_disk_as_reward=False,
         render=True,
         initial_walls=0,
         get_results=False):
    print('protagonist: {}'.format(protagonist_agent_type))
    print('opponent: {}'.format(opponent_agent_type))

    protagonist_policy = create_policy(
        policy_type=protagonist_agent_type,
        board_size=board_size,
        seed=rand_seed,
        search_depth=protagonist_search_depth,
        learned=protagonist_learned,
        q_values=protagonist_Qvalues,
        epi=protagonist_learned_games)
    opponent_policy = create_policy(
        policy_type=opponent_agent_type,
        board_size=board_size,
        seed=rand_seed,
        search_depth=opponent_search_depth,
        learned=opponent_learned,
        q_values=opponent_Qvalues,
        epi=opponent_learned_games)

    if protagonist == 1:
        white_policy = protagonist_policy
        black_policy = opponent_policy
    else:
        white_policy = opponent_policy
        black_policy = protagonist_policy

    if opponent_agent_type == 'human':
        render_in_step = True
    else:
        render_in_step = False

    env = othello.OthelloEnv(white_policy=white_policy,
                             black_policy=black_policy,
                             protagonist=protagonist,
                             board_size=board_size,
                             seed=rand_seed,
                             initial_rand_steps=env_init_rand_steps,
                             num_disk_as_reward=num_disk_as_reward,
                             render_in_step=render_in_step and render,
                             initial_walls=initial_walls)

    win_cnts = draw_cnts = lose_cnts = no_game_cnts = 0
    last_100_episode_rewards = deque(maxlen=100)
    last_100_episode_wins = deque(maxlen=100)
    last_100_episode_loses = deque(maxlen=100)
    last_100_episode_draws = deque(maxlen=100)
    
    # For drawing result graph
    game_idx = [] # games
    game_reward = [] # rewards

    for i in range(num_rounds):
        print('Episode {}'.format(i + 1))
        obs, check_no_game = env.reset()
        protagonist_policy.reset(env)

        if render:
            env.render()
        if check_no_game:
            no_game_cnts += 1
            print('No possible moves for some party at their first turn.')
            print('This game is invalid.')
            print('-' * 35)
            continue
        done = False
        while not done:
            action = protagonist_policy.get_action(obs)
            new_obs, reward, done, _ = env.step(action)
            protagonist_policy.step(obs, action, reward, new_obs, done)
            obs = new_obs
            if render:
                env.render()
            if done:
                print('reward={}'.format(reward))
                last_100_episode_rewards.append(reward)
                avg_reward = sum(last_100_episode_rewards) / len(last_100_episode_rewards)
                print("\raverage reward of last 100 games : {}\n".format(avg_reward), end="")

                if (i+1) % max(1, num_rounds // 100) == 0:
                    game_idx.append(i+1)
                    game_reward.append(avg_reward)
                
                if num_disk_as_reward:
                    total_disks = board_size ** 2
                    if protagonist == 1:
                        white_cnts = reward
                        black_cnts = total_disks - white_cnts
                    else:
                        black_cnts = reward
                        white_cnts = total_disks - black_cnts

                    if white_cnts > black_cnts:
                        win_cnts += 1
                        last_100_episode_wins.append(1)
                        last_100_episode_loses.append(0)
                        last_100_episode_draws.append(0)
                    elif white_cnts == black_cnts:
                        draw_cnts += 1
                        last_100_episode_wins.append(0)
                        last_100_episode_loses.append(0)
                        last_100_episode_draws.append(1)
                    else:
                        lose_cnts += 1
                        last_100_episode_wins.append(0)
                        last_100_episode_loses.append(1)
                        last_100_episode_draws.append(0)
                else:
                    if reward == 1:
                        win_cnts += 1
                        last_100_episode_wins.append(1)
                        last_100_episode_loses.append(0)
                        last_100_episode_draws.append(0)
                    elif reward == 0:
                        draw_cnts += 1
                        last_100_episode_wins.append(0)
                        last_100_episode_loses.append(0)
                        last_100_episode_draws.append(1)
                    else:
                        lose_cnts += 1
                        last_100_episode_wins.append(0)
                        last_100_episode_loses.append(1)
                        last_100_episode_draws.append(0)
                wins = sum(last_100_episode_wins)
                loses = sum(last_100_episode_loses)
                draws = sum(last_100_episode_draws)
                print('Recent 100 Games - #Wins : {}, #Draws: {}, #Loses: {}'.format(wins, draws, loses))
                print('-' * 35)
    protagonist_Qvalues = protagonist_policy.q_values
    protagonist_num_episodes = protagonist_policy.num_episodes
    opponent_Qvalues = env.opponent.q_values
    opponent_num_episodes = env.opponent.num_episodes
    print('<< TOTAL RESULTS >>')
    print('#Wins: {}, #Draws: {}, #Loses: {}, #No_games: {}'.format(
        win_cnts, draw_cnts, lose_cnts, no_game_cnts))
    if get_results:
        data = {'X': game_idx, 'Y': game_reward}
        df = pd.DataFrame(data)
        excel_file_name = '{} vs {}.xlsx'.format(protagonist_agent_type, opponent_agent_type)
        df.to_excel(excel_file_name, index=False, engine='openpyxl')
    env.close()
    return protagonist_Qvalues, protagonist_num_episodes, opponent_Qvalues, opponent_num_episodes

if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--rand-seed', default=0, type=int)
    parser.add_argument('--no-render', default=False, action='store_true')
    parser.add_argument('--get-results', default=False, action='store_true')
    args, _ = parser.parse_known_args()

    protagonist_learned = False
    protagonist_Qvalues = None
    protagonist_num_episodes = 0
    opponent_learned = False
    opponent_Qvalues = None
    opponent_num_episodes = 0
    policies_list = ['human', 'rand', 'greedy', 'minimax', 'DQN', 'DQN']
    board_size = 8
    num_obstacle = 0
    init_rand_steps = 0
    num_rounds = 100
    num_disk_as_reward = False
    protagonist_search_depth = 1
    opponent_search_depth = 1
    option_check_code = True
    while True:
        if option_check_code:
            print("<< Game Setting >>")
            board_size = int(input("Board Size(min : 4) : "))
            num_obstacle = int(input("Number of obstacles : "))
            init_rand_steps = int(input("Initial Random Steps : "))
            num_rounds = int(input("Number of Rounds : "))
            print("Reward Setting - 1. Win/Lose   2. Num of disks")
            reward_code = int(input("Select : "))
            num_disk_as_reward = True if reward_code == 2 else False
            print("<< Players Setting >>")
            print("1. human")
            print("2. Random Policy")
            print("3. Greedy Policy")
            print("4. Minimax Policy")
            print("5. DQN(Before Learning)")
            print("6. DQN(After Learning)")
            protagonist_policy_code = int(input("Protagonist Player's Policy : "))
            protagonist_agent_type = policies_list[protagonist_policy_code - 1]
            if protagonist_policy_code == 4:
                protagonist_search_depth = int(input("Search Depth : "))
            if protagonist_policy_code == 5:
                protagonist_learned = False
            if protagonist_policy_code == 6:
                protagonist_learned = True
            opponent_policy_code = int(input("Opponent Player's Policy : "))
            opponent_agent_type = policies_list[opponent_policy_code - 1]
            if opponent_policy_code == 4:
                opponent_search_depth = int(input("Search Depth : "))
            if opponent_policy_code == 5:
                opponent_learned = False
            if opponent_policy_code == 6:
                opponent_learned = True
            disk_color_code = input("Protagonist Player's Color(Black(B)/White(W)) : ")
            protagonist = 1 if disk_color_code == 'W' or disk_color_code == 'w' else -1
            if not protagonist_learned:
                protagonist_Qvalues = None
                protagonist_num_episodes = 0
            if not opponent_learned:
                opponent_Qvalues = None
                opponent_num_episodes = 0
        

        # Run test plays.
        
        protagonist_Qvalues, protagonist_num_episodes, opponent_Qvalues, opponent_num_episodes = play(protagonist=protagonist,
                                                                                                      protagonist_agent_type=protagonist_agent_type,
                                                                                                      opponent_agent_type=opponent_agent_type,
                                                                                                      board_size=board_size,
                                                                                                      num_rounds=num_rounds,
                                                                                                      protagonist_search_depth=protagonist_search_depth,
                                                                                                      opponent_search_depth=opponent_search_depth,
                                                                                                      protagonist_learned=protagonist_learned,
                                                                                                      protagonist_Qvalues=protagonist_Qvalues,
                                                                                                      protagonist_learned_games=protagonist_num_episodes,
                                                                                                      opponent_learned=opponent_learned,
                                                                                                      opponent_Qvalues=opponent_Qvalues,
                                                                                                      opponent_learned_games=opponent_num_episodes,
                                                                                                      rand_seed=args.rand_seed,
                                                                                                      env_init_rand_steps=init_rand_steps,
                                                                                                      num_disk_as_reward=num_disk_as_reward,
                                                                                                      render=not args.no_render,
                                                                                                      initial_walls=num_obstacle,
                                                                                                      get_results=args.get_results)
        continue_check = input("\nDo you want to continue the game?(Y/n) : ")
        if continue_check == 'n':
            break
        option_check = input("Do you want to change the game option?(Y/n) : ")
        option_check_code = False if option_check == 'n' else True

