import gym
import numpy as np
import torch
import torch.optim as optim

import gym_anton
from gym_anton.model import TransformerQnet
from gym_anton.utils import ReplayBuffer
from gym_anton.utils import dqn_trainer
from gym_anton.utils import TensorboardManager
from gym_anton.utils import StatsManager


def main():
    env = gym.make("spot-v0", df=df, window_size=window_size)
    q = TransformerQnet(d_model=8, nhead=2, num_layers=2, num_seq=window_size)
    q_target = TransformerQnet(d_model=8, nhead=2, num_layers=2, num_seq=window_size)
    q_target.load_state_dict(q.state_dict())

    q.to(device)
    q_target.to(device)

    memory = ReplayBuffer(buffer_limit=buffer_limit, device=device)
    optimzer = optim.Adam(q.parameters(), lr=learning_rate)

    # Monitoring metric

    terminate = False
    n_epi = 0

    # main loop
    while not terminate:
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 2000))
        s = env.reset()
        done = False

        while not done:
            a = q.sample_action(
                torch.from_numpy(np.expand_dims(s, axis=0)).float().to(device), epsilon
            )
            s_prime, r, done, terminate = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r, s_prime, done_mask))
            s = s_prime

            if done:
                epi_info = env.get_episode_info()
                st_manager.add_stats(epi_info)
                break

        if memory.size() > 5000:
            avg_loss = dqn_trainer(q, q_target, memory, optimzer, gamma, batch_size=batch_size)

            # info = {"Avg Loss": avg_loss, "Cumulative Reward": score}
            # tb_manager.add(n_epi, info)

        if n_epi % target_update_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())  # update target network

        if n_epi % log_interval == 0 and n_epi != 0:
            st_manager.reset()

        n_epi += 1

    env.close()


if __name__ == "__main__":
    # Hyperparameters
    learning_rate = 0.0005
    gamma = 0.98
    buffer_limit = 20000
    batch_size = 32
    log_interval = 100
    target_update_interval = 200
    window_size = 24
    if torch.cuda.is_available():
        print("CUDA is available")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # load dataset
    df = gym_anton.datasets.BTCUSDT_10M.copy()

    # Manager
    tb_manager = TensorboardManager()
    st_manager = StatsManager()

    main()
