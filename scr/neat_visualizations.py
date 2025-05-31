from __future__ import print_function

import copy
import warnings
import graphviz
import matplotlib.pyplot as plt
import numpy as np


def plot_fitness_statistics(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
    """
    Plot the population's average and best fitness over generations.
    
    Parameters:
        statistics: neat.StatisticsReporter object.
        ylog (bool): Use logarithmic scale on Y-axis.
        view (bool): Whether to display the plot after saving.
        filename (str): Output file name for the saved plot.
    """
    if plt is None:
        warnings.warn("Plotting not available: matplotlib is not installed.")
        return

    generations = range(len(statistics.most_fit_genomes))
    best_fitness = [genome.fitness for genome in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generations, avg_fitness, 'b-', label="Average")
    plt.plot(generations, avg_fitness - stdev_fitness, 'g-.', label="-1 SD")
    plt.plot(generations, avg_fitness + stdev_fitness, 'g-.', label="+1 SD")
    plt.plot(generations, best_fitness, 'r-', label="Best")

    plt.title("Population Average and Best Fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")

    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


def plot_spiking_neuron_activity(spikes, view=False, filename=None, title=None):
    """
    Plot the internal state of an Izhikevich spiking neuron.
    
    Parameters:
        spikes (list): List of (time, input, v, u, fired) tuples.
        view (bool): Display the plot window after saving.
        filename (str): Save to file if provided.
        title (str): Custom title.
        
    Returns:
        Matplotlib Figure object.
    """
    t_vals = [t for t, I, v, u, f in spikes]
    v_vals = [v for t, I, v, u, f in spikes]
    u_vals = [u for t, I, v, u, f in spikes]
    I_vals = [I for t, I, v, u, f in spikes]
    f_vals = [f for t, I, v, u, f in spikes]

    fig = plt.figure()

    plt.subplot(4, 1, 1)
    plt.ylabel("Membrane Potential (mV)")
    plt.xlabel("Time (ms)")
    plt.grid()
    plt.plot(t_vals, v_vals, "g-")
    plt.title(f"Izhikevich Neuron Activity ({title})" if title else "Izhikevich Neuron Activity")

    plt.subplot(4, 1, 2)
    plt.ylabel("Firing")
    plt.xlabel("Time (ms)")
    plt.grid()
    plt.plot(t_vals, f_vals, "r-")

    plt.subplot(4, 1, 3)
    plt.ylabel("Recovery Variable (u)")
    plt.xlabel("Time (ms)")
    plt.grid()
    plt.plot(t_vals, u_vals, "r-")

    plt.subplot(4, 1, 4)
    plt.ylabel("Input Current (I)")
    plt.xlabel("Time (ms)")
    plt.grid()
    plt.plot(t_vals, I_vals, "r-o")

    if filename:
        plt.savefig(filename)

    if view:
        plt.show()
        plt.close()
        fig = None

    return fig


def plot_species_distribution(statistics, view=False, filename='speciation.svg'):
    """
    Plot the distribution of species sizes over generations.
    
    Parameters:
        statistics: neat.StatisticsReporter object.
        view (bool): Show plot window after saving.
        filename (str): Output file name for the saved plot.
    """
    if plt is None:
        warnings.warn("Plotting not available: matplotlib is not installed.")
        return

    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation Over Generations")
    plt.xlabel("Generations")
    plt.ylabel("Individuals per Species")

    plt.savefig(filename)

    if view:
        plt.show()

    plt.close()


def draw_neural_network(config, genome, view=False, filename=None, node_names=None,
                        show_disabled=True, prune_unused=False, node_colors=None, fmt='svg'):
    """
    Visualize a neural network genome using Graphviz.
    
    Parameters:
        config: NEAT configuration.
        genome: Genome object.
        view (bool): Open viewer after saving.
        filename (str): File to save output (without extension).
        node_names (dict): Mapping of node key to name.
        show_disabled (bool): Display disabled connections.
        prune_unused (bool): Remove unused nodes.
        node_colors (dict): Node color map.
        fmt (str): Output file format (e.g. 'svg', 'png').
        
    Returns:
        Graphviz Digraph object.
    """
    if graphviz is None:
        warnings.warn("Graphviz not installed; cannot render network.")
        return

    node_names = node_names or {}
    node_colors = node_colors or {}

    node_attr = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'
    }

    dot = graphviz.Digraph(format=fmt, node_attr=node_attr)

    # Input nodes
    input_keys = config.genome_config.input_keys
    for k in input_keys:
        name = node_names.get(k, str(k))
        dot.node(name, _attributes={
            'style': 'filled',
            'shape': 'box',
            'fillcolor': node_colors.get(k, 'lightgray')
        })

    # Output nodes
    output_keys = config.genome_config.output_keys
    for k in output_keys:
        name = node_names.get(k, str(k))
        dot.node(name, _attributes={
            'style': 'filled',
            'fillcolor': node_colors.get(k, 'lightblue')
        })

    # Determine which nodes to draw
    if prune_unused:
        used_nodes = set(output_keys)
        connections = {(c.in_node_id, c.out_node_id) for c in genome.connections.values() if c.enabled or show_disabled}
        pending = set(output_keys)
        while pending:
            new_pending = set()
            for a, b in connections:
                if b in pending and a not in used_nodes:
                    used_nodes.add(a)
                    new_pending.add(a)
            pending = new_pending
    else:
        used_nodes = set(genome.nodes.keys())

    # Hidden nodes
    for node in used_nodes:
        if node in input_keys or node in output_keys:
            continue
        dot.node(str(node), _attributes={
            'style': 'filled',
            'fillcolor': node_colors.get(node, 'white')
        })

    # Draw connections
    for connection in genome.connections.values():
        if connection.enabled or show_disabled:
            input_node, output_node = connection.key
            input_name = node_names.get(input_node, str(input_node))
            output_name = node_names.get(output_node, str(output_node))
            style = 'solid' if connection.enabled else 'dotted'
            color = 'green' if connection.weight > 0 else 'red'
            width = str(0.1 + abs(connection.weight / 5.0))
            dot.edge(input_name, output_name, _attributes={
                'style': style,
                'color': color,
                'penwidth': width
            })

    dot.render(filename, view=view)
    return dot
