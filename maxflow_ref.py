import copy

import numpy as np

if __name__ == '__main__':
    adj_capacities = np.array([[0, 7, 0, 0],
           [0, 0, 6, 0],
           [0, 0, 0, 8],
           [9, 0, 0, 0]])

    # this is just the running capacity
    residuals = copy.deepcopy(adj_capacities)

    node_labels = np.array([0, 0, 0, 0])
    node_labels[0] = len(node_labels)

    
    # f=c -> r=c-f=0
    residuals[0, :] = 0
    residuals[:, 0] += adj_capacities[0, :]

    node_excess = np.array(adj_capacities[0, :])

    while np.any(node_excess[1:-1] > 0):
        # do not iterate over source or sink
        for node_id in range(1, len(adj_capacities) - 1):
            push_eligible = residuals[node_id] > 0
            assert len(push_eligible) > 0 # if excess > 0 then there should be outgoing connections

            lowest_label_index = np.where(push_eligible, node_labels, np.inf).argmin()
            lowest_label = node_labels[lowest_label_index]

            if node_labels[node_id] > lowest_label:
                # push!
                diff = np.minimum(node_excess[node_id], residuals[node_id, lowest_label_index])
                print(f"Node {node_id} pushing to {lowest_label_index} with diff {diff}")
                residuals[node_id, lowest_label_index] -= diff
                residuals[lowest_label_index, node_id] += diff
                node_excess[node_id] -= diff
                node_excess[lowest_label_index] += diff
            else:
                # relabel!
                print(f"Node {node_id} re-labelling")
                node_labels[node_id] = lowest_label + 1


    assert np.sum((adj_capacities - residuals) == 6)
