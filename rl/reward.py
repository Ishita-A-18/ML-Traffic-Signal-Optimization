def compute_reward(metrics, prev_metrics):
    q = metrics["queue"]
    w = metrics["waiting"]

    reward = -(0.7 * q + 0.3 * w)

    return reward
