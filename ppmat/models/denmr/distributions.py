import paddle


class DistributionNodes:
    def __init__(self, histogram):
        """Compute the distribution of the number of nodes in the dataset,
            and sample from this distribution.
        historgram: dict. The keys are num_nodes, the values are counts
        """
        if type(histogram) == dict:
            max_n_nodes = max(histogram.keys())
            prob = paddle.zeros(shape=max_n_nodes + 1)
            for num_nodes, count in histogram.items():
                prob[num_nodes] = count
        else:
            prob = histogram
        self.prob = prob / prob.sum()
        self.m = paddle.distribution.Categorical(prob)

    def sample_n(self, n_samples):
        idx = self.m.sample((n_samples,))
        return idx

    def log_prob(self, batch_n_nodes):
        assert len(tuple(batch_n_nodes.shape)) == 1
        p = self.prob.to(batch_n_nodes.place)
        probas = p[batch_n_nodes]
        log_p = paddle.log(x=probas + 1e-30)
        return log_p
