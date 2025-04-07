
class GreedyPolicy:
    """A simple greedy policy that selects the node with the minimal 
    predicted total time (transmission + computation)."""

    def act(self, env, task):
        """
        Greedily choose the node that yields the lowest estimated total latency.

        Args:
            env (Env): The environment (for accessing node data, if needed).
            task (Task): The current task to be scheduled/offloaded.

        Returns:
            int: The selected action (index of the chosen node).
        """
        best_node = None
        best_latency = float('inf')

        # Iterate through all possible node IDs in the environment
        for node_id in range(len(env.scenario.node_id2name)):
            node_name = env.scenario.node_id2name[node_id]

            cpu_speed = env.scenario.get_node(node_name).free_cpu_freq
            transmission_time = task.task_size / task.trans_bit_rate  # seconds
            computation_time = (task.task_size * task.cycles_per_bit) /( cpu_speed + 1)
            
            total_time = transmission_time + computation_time

            # Greedy choice: pick the node with the lowest total_time
            if total_time < best_latency:
                best_latency = total_time
                best_node = node_id

        return best_node

    def act_greed(self, env, task):
        """
        Greedily choose the node that yields the lowest estimated total latency, considering trust.

        Args:
            env (Env): The environment (for accessing node data, if needed).
            task (Task): The current task to be scheduled/offloaded.

        Returns:
            int: The selected action (index of the chosen node).
        """
        best_node = None
        best_latency = float('inf')
        best_trust = float('-inf')  # Track the highest trust for tie-breaking

        # Iterate through all possible node IDs in the environment
        for node_id in range(len(env.scenario.node_id2name)):
            node = env.scenario.get_node(f'n{node_id}')

            cpu_speed = node.free_cpu_freq
            transmission_time = task.task_size / task.trans_bit_rate  # seconds
            computation_time = (task.task_size * task.cycles_per_bit) / (cpu_speed + 1)
            
            total_time = transmission_time + computation_time
            trust = env.global_trust[node]  # Fetch trust, default to 0 if not found

            # Greedy choice: prioritize lowest total_time, break ties with higher trust
            if total_time < best_latency or (total_time == best_latency and trust > best_trust):
                best_latency = total_time
                best_trust = trust
                best_node = node_id

        return best_node
