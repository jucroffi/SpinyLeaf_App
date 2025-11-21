def directed_cycle(nodes, edges):
    nodes_left = set(nodes)
    while nodes_left:
        node = nodes_left.pop()
        stack = [node]
        while stack:
            top = stack[-1]
            for node in edges[top]:
                if node in stack:
                    return stack[stack.index(node):]
                if node in nodes_left:
                    stack.append(node)
                    nodes_left.remove(node)
                    break
            else:
                node = stack.pop()
    return None


def isolated_nodes(nodes, edges):
    # Only the nodes that don't point anywhere
    return list(node for node in nodes if not len(edges[node]))


def evaluation_order(nodes, edges):
    cycle = directed_cycle(nodes, edges)
    if cycle:
        raise ValueError(
            "Circular dependency is not allowed. Cycle found: %s"
            % '->'.join([""] + cycle + [""])
        )
    left = set(nodes)
    edges_ = dict(edges.iteritems())
    while left:
        for node in isolated_nodes(left, edges_):
            left.remove(node)
            yield node

        # Remove edges of nodes that point to leaves only
        for node in left:
            total_out_degree = sum(len(edges_[target]) for target in edges_[node])
            if total_out_degree == 0:
                edges_[node] = []
