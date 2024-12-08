#include "thrust/device_vector.h"
#include <thrust/logical.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include "thrust/sequence.h"
#include "thrust/device_new.h"
#include "ranges"

constexpr bool DEBUG = false;

struct label_if_res_functor {
    __host__ __device__
    // get the lowest label from residuals that is >0, otherwise inf to rule it out
    thrust::tuple<int, int> operator()(const thrust::tuple<int, int, int> &t) const {
        if (thrust::get<1>(t) > 0)
            return thrust::make_tuple(thrust::get<0>(t), thrust::get<2>(t));
        return thrust::make_tuple(-1, INT_MAX);
    }
};

struct excess_remains_functor {
    __host__ __device__
    bool operator()(const int node_excess) const {
        return node_excess > 0;
    }
};

struct pair_argmin_functor {
    __host__ __device__
    thrust::tuple<int, int> operator()(thrust::tuple<int, int> a, thrust::tuple<int, int> b) const {
       if (thrust::get<1>(a) < thrust::get<1>(b)) {
            return a;
        }
        return b;
    }
};

struct do_everything_functor {
    int *residuals; // residuals (N,N), rest N
    int *node_excess;
    int *node_labels;
    int N;

    __device__
    void operator()(const int node_id) const {
        const thrust::counting_iterator<int> first(0);
        const thrust::counting_iterator<int> last(N);

        const auto begin = make_zip_iterator(make_tuple(first, residuals + N * node_id, node_labels));
        const auto end = make_zip_iterator(make_tuple(last, residuals + N * (node_id + 1), node_labels + N));

        auto min_label_tuple = thrust::transform_reduce(thrust::device,
                                                  begin, end, label_if_res_functor(), thrust::make_tuple(-1, INT_MAX),
                                                  pair_argmin_functor());
        const auto min_label_idx = thrust::get<0>(min_label_tuple);
        const auto min_label_value = thrust::get<1>(min_label_tuple);

        if (DEBUG)
            printf("Node %d found min. label: %d, idx: %d\n", node_id, min_label_value, min_label_idx);

        if (min_label_value == INT_MAX) return;

        if (node_labels[node_id] > min_label_value) {
            const auto diff = thrust::min(node_excess[node_id], residuals[N * node_id + min_label_idx]);
            atomicSub(residuals + N * node_id + min_label_idx, diff);
            atomicAdd(residuals + N * min_label_idx + node_id, diff);
            if (DEBUG)
            printf("Pushing from %d to %d with diff %d\n", node_id, min_label_idx, diff);

            atomicSub(node_excess + node_id, diff);
            atomicAdd(node_excess + min_label_idx, diff);
        } else {
            if (DEBUG)
            printf("Elevate on %d\n", node_id);
            node_labels[node_id] = min_label_value + 1;
        }
    }
};

int main() {
    // adjacency matrix of initial capacities
    const std::vector<std::vector<int> > adj_c = {
        {0, 7, 0, 0},
        {0, 0, 6, 0},
        {0, 0, 0, 8},
        {9, 0, 0, 0}
    };

    const int N = adj_c.size();
    auto residuals = adj_c;

    auto node_levels = std::vector<int>(adj_c.size());
    node_levels[0] = static_cast<int>(node_levels.size());

    // f=c -> r=c-f=0
    std::ranges::fill(residuals[0], 0);
    for (int i = 0; i < residuals.size(); ++i) {
        residuals[i][0] = adj_c[0][i];
    }

    auto node_excess = std::vector<int>(adj_c.size());
    std::ranges::copy(adj_c[0], node_excess.begin());
    // flatten residuals using ranges view join
    auto residuals_flat = std::ranges::join_view(residuals);

    auto residuals_cuda = thrust::device_vector<int>(residuals_flat.begin(), residuals_flat.end());
    auto node_levels_cuda = thrust::device_vector<int>(node_levels.begin(), node_levels.end());
    auto node_excess_cuda = thrust::device_vector<int>(node_excess.begin(), node_excess.end());

    // todo: figure out fancier way
    auto node_indices_cuda = thrust::device_vector<int>(node_levels_cuda.size());
    sequence(node_indices_cuda.begin(), node_indices_cuda.end());

    bool excess_remains = true;
    const auto residuals_ptr = raw_pointer_cast(residuals_cuda.data());
    const auto node_excess_ptr = raw_pointer_cast(node_excess_cuda.data());
    const auto node_levels_ptr = raw_pointer_cast(node_levels_cuda.data());

    while (excess_remains) {
        for_each(thrust::device, node_indices_cuda.begin() + 1,  node_indices_cuda.end() - 1,
            do_everything_functor{
            residuals_ptr,
            node_excess_ptr,
            node_levels_ptr,
            N
        });

        excess_remains = any_of(thrust::device, node_excess_cuda.begin() + 1, node_excess_cuda.end() - 1, excess_remains_functor());
    }

    int result = -1;
    copy(node_excess_cuda.end() - 1, node_excess_cuda.end(), &result);
    printf("Result: %d\n", result);

    return 0;
}
