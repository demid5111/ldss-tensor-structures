import networkx as nx

from src.layers.ilayer import ILayer


class FeedForwardNetwork:
    def __init__(self):
        self.input_layers = []
        self.output_layers = []
        self.hidden_layers = []
        self.inputs = []
        self.graph = nx.DiGraph()
        self.counter = 1

    def add_input_layer(self, l):
        self.add_layer(l=l)
        self.input_layers.append(l)

    def add_layer(self, l: ILayer, parents=(), is_output=False):
        new_id = self.unique_id()
        self.graph.add_node(new_id, **dict(layer_ref=l))
        l.set_id(new_id)
        if is_output:
            self.output_layers.append(l)
        if parents:
            self.graph.add_edges_from([(p_id, new_id) for p_id in parents])

    def add_input(self, inp):
        self.inputs.append(inp)

    def forward(self):
        for layer_id in nx.topological_sort(self.graph):
            parents = self.graph.predecessors(layer_id)
            self.graph.nodes[layer_id]['layer_ref'].forward([self.graph.nodes[p_id]['layer_ref'].output for p_id in parents])

    def unique_id(self):
        self.counter += 1
        return self.counter

    def fill_input(self, id, data):
        self.graph.nodes[id]['layer_ref'].input_data = data

    def dump_structure(self):
        import pygraphviz
        from networkx.drawing.nx_agraph import write_dot

        print("using package pygraphviz")
        write_dot(self.graph, "grid.dot")

    def outputs(self):
        outs = [x for x in self.graph.nodes() if self.graph.out_degree(x) == 0 and self.graph.in_degree(x) >= 1]
        return [self.graph.nodes[o_id]['layer_ref'].output for o_id in outs]