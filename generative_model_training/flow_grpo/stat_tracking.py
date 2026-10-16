import numpy as np


class PerPromptStatTracker:
    def __init__(self, global_std=False, ban_std_thres=0.05, ban_mean_thres=0.9):
        self.global_std = global_std
        self.stats = {}
        self.history_prompts = set()

        # Banned prompt
        self.ban_std_thres = ban_std_thres
        self.ban_mean_thres = ban_mean_thres
        self.banned_prompts = set()

    # exp reward is for rwr
    def update(self, prompts, rewards, exp=False, valid_mask=None):
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        valid = np.ones(len(prompts), dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool).copy()
        if valid.shape != (len(prompts),) or rewards.shape[0] != len(prompts):
            raise ValueError("Prompts, rewards and validity mask must have the same batch length")
        if len(prompts):
            valid &= np.isfinite(rewards).reshape(len(prompts), -1).all(axis=1)
        unique = np.unique(prompts[valid])
        advantages = np.zeros_like(rewards)
        stds = np.zeros_like(rewards)
        means = np.zeros_like(rewards)

        for prompt in unique:
            prompt_rewards = rewards[(prompts == prompt) & valid]
            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].extend(prompt_rewards)
            self.history_prompts.add(hash(prompt))  # Add hash of prompt to history_prompts
        for prompt in unique:
            history = np.stack(self.stats[prompt])
            group_mask = (prompts == prompt) & valid
            prompt_rewards = rewards[group_mask]
            mean = np.mean(history, axis=0, keepdims=True)

            if self.global_std:
                std = np.std(rewards[valid], axis=0, keepdims=True) + 1e-4
            else:
                std = np.std(history, axis=0, keepdims=True) + 1e-4

            prompt_std = np.std(history, axis=0, keepdims=True).mean()
            prompt_mean = np.mean(history, axis=0, keepdims=True).mean()

            if prompt_std < self.ban_std_thres and prompt_mean > self.ban_mean_thres:
                self.banned_prompts.add(prompt)

            advantages[group_mask] = (prompt_rewards - mean) / std
            stds[group_mask] = prompt_std
            means[group_mask] = mean

        return advantages, stds, means

    def get_stats(self):
        avg_group_size = sum(len(v) for v in self.stats.values()) / len(self.stats) if self.stats else 0
        history_prompts = len(self.history_prompts)
        return avg_group_size, history_prompts

    def clear(self):
        self.stats = {}

    def get_mean_of_top_rewards(self, top_percentage):
        if not self.stats:
            return 0.0

        assert 0 <= top_percentage <= 100

        per_prompt_top_means = []
        for prompt_rewards in self.stats.values():
            if isinstance(prompt_rewards, list):
                rewards = np.array(prompt_rewards)
            else:
                rewards = prompt_rewards

            if rewards.size == 0:
                continue

            if top_percentage == 100:
                per_prompt_top_means.append(np.mean(rewards))
                continue

            lower_bound_percentile = 100 - top_percentage
            threshold = np.percentile(rewards, lower_bound_percentile)

            top_rewards = rewards[rewards >= threshold]

            if top_rewards.size > 0:
                per_prompt_top_means.append(np.mean(top_rewards))

        if not per_prompt_top_means:
            return 0.0

        return np.mean(per_prompt_top_means)


def main():
    tracker = PerPromptStatTracker()
    prompts = ["a", "b", "a", "c", "b", "a"]
    rewards = [1, 2, 3, 4, 5, 6]
    advantages = tracker.update(prompts, rewards)
    print("Advantages:", advantages)
    avg_group_size, history_prompts = tracker.get_stats()
    print("Average Group Size:", avg_group_size)
    print("History Prompts:", history_prompts)
    tracker.clear()
    print("Stats after clear:", tracker.stats)


if __name__ == "__main__":
    main()
