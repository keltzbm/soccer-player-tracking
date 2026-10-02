# Ornstein-Uhlenbeck (OU) process
# dX_t = theta * (mu - X_t) * dt + sigma * dW_t
# X(t + dt) = (1 - theta * dt) * X(t) + theta * mu * dt + sigma * sqrt(dt) * N(0, 1)


# (1 - theta * dt) * mu + theta * mu * dt + sigma * sqrt(dt) * N(0, 1)
# mu + sigma * sqrt(dt) * N(0, 1)
import numpy as np

n_players = 11
time = 45


# 10 Hz
dt = 10
n_steps = 60 * time * dt 

sigmas = np.array([0.1, 0.5, 0.4, 0.4, 0.5, 0.8, 0.8, 0.8, 0.9, 1, 0.9])

gps_means = np.zeros(n_players)
gps_sigmas = np.random.uniform(1, 5, size=n_players)

gps_noise = np.random.normal(loc=gps_means, scale=gps_sigmas, size=(n_steps, n_players))

# Ornstein-Uhlenbeck (OU) process
# dX_t = theta * (mu - X_t) * dt + sigma * dW_t

positions = np.zeros((n_players, n_steps, 2))
noise = np.random.normal(loc=0, scale=1, size=(n_steps, n_players))

theta = 1 / 5
home = theta * positions[:, 0, :] * dt

for i in range(1, n_steps):
    noise = sigmas * np.sqrt(dt) * np.random.normal(loc=0, scale=1, size=(2, n_players))
    positions[:, i, :] = (1 - theta * dt) * positions[:, i - 1, :] + home + noise.T 


