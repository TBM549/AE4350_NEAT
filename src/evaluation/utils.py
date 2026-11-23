import neat as neat
import numpy as np
from neat.activations import tanh_activation
from neat.aggregations import sum_aggregation


def activate_batch(net, observations):
    observations = np.asarray(observations)
    if observations.ndim != 2 or observations.shape[1] != len(net.input_nodes):
        raise ValueError(
            f"Network expects {len(net.input_nodes)} inputs per observation, "
            f"but received shape {observations.shape}. Match NEAT num_inputs "
            "to the environment observation encoding (9 Cartesian, 11 current or 12 legacy magnitude/direction)."
        )
    # Other activation or aggregation functions keep using NEAT's evaluator.
    if any(act != tanh_activation or agg != sum_aggregation
           for _, act, agg, _, _, _ in net.node_evals):
        return np.array([net.activate(obs) for obs in observations])

    values = {node: np.zeros(len(observations)) for node in net.output_nodes}
    for node, column in zip(net.input_nodes, observations.T):
        values[node] = column

    for node, _, _, bias, response, links in net.node_evals:
        total = sum(values[source] * weight for source, weight in links)
        # Match NEAT's tanh scaling and clipping.
        values[node] = np.broadcast_to(
            np.tanh(np.clip(2.5 * (bias + response * total), -60.0, 60.0)),
            (len(observations),),
        )

    return np.column_stack([values[node] for node in net.output_nodes])


def evaluate_controller(genome, config, env, scenarios=None):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    observations, dones = env.reset(scenarios=scenarios)

    while not np.all(dones):
        accel_cmds = np.zeros((len(observations), len(net.output_nodes)), dtype=np.float32)
        accel_cmds[~dones] = activate_batch(net, observations[~dones])
        observations, dones = env.step(accel_cmds)

    fitness = env.calculate_fitness()

    return float(np.mean(fitness))
