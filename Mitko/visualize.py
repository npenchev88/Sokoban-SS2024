
import matplotlib.pyplot as plt
import graphviz
import warnings
import numpy as np

def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.figure(figsize=(8, 6))
    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    plt.savefig(filename)
    if view:
        plt.show()
    plt.close()

def plot_species(statistics, view=False, filename='speciation.svg'):
    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    plt.figure(figsize=(8, 6))
    plt.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.xlabel("Generations")
    plt.ylabel("Species Size")
    plt.savefig(filename)
    if view:
        plt.show()
    plt.close()

def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False, node_colors=None, fmt='svg'):
    if graphviz is None:
        warnings.warn("This display is not available due to a missing dependency: graphviz")
        return

    if node_names is None:
        node_names = {}

    if node_colors is None:
        node_colors = {}

    # Attributes for network nodes
    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box'}
        input_attrs['fillcolor'] = node_colors.get(k, 'lightgray')
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled'}
        node_attrs['fillcolor'] = node_colors.get(k, 'lightblue')
        dot.node(name, _attributes=node_attrs)

    used_nodes = set(genome.nodes.keys())
    if prune_unused:
        used_nodes = set(config.genome_config.output_keys)
        connections = {cg.key for cg in genome.connections.values() if cg.enabled}

        while True:
            new_used_nodes = used_nodes.copy()
            for a, b in connections:
                if b in used_nodes:
                    new_used_nodes.add(a)
            if len(new_used_nodes) == len(used_nodes):
                break
            used_nodes = new_used_nodes

    for node, cg in genome.nodes.items():
        if node in inputs or node in outputs or node in used_nodes:
            attrs = {'style': 'filled'}
            attrs['fillcolor'] = node_colors.get(node, 'white')
            dot.node(node_names.get(node, str(node)), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            input_node, output_node = cg.key
            if prune_unused and (input_node not in used_nodes or output_node not in used_nodes):
                continue
            a = node_names.get(input_node, str(input_node))
            b = node_names.get(output_node, str(output_node))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.enabled else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)
