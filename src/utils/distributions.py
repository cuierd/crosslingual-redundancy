import torch
import torch.distributions as dist

class BimodalGaussianDistribution(dist.Distribution):
    def __init__(self, mean1, std1, weight1, mean2, std2):
        self.normal1 = dist.Normal(mean1, std1)
        self.normal2 = dist.Normal(mean2, std2)
        self.weight1 = weight1
        self.weight2 = 1 - weight1
        super(BimodalGaussianDistribution, self).__init__()

    def sample(self, sample_shape=torch.Size()):
        shape = torch.Size(sample_shape)
        # Use the Bernoulli distribution to decide which component to sample from
        choose_component = dist.Bernoulli(torch.tensor(self.weight1)).sample(shape)
        samples1 = self.normal1.sample(shape)
        samples2 = self.normal2.sample(shape)
        # Combine samples based on the chosen component
        samples = choose_component * samples1 + (1 - choose_component) * samples2
        return samples

    def log_prob(self, value):
        prob1 = torch.exp(self.normal1.log_prob(value))
        prob2 = torch.exp(self.normal2.log_prob(value))
        # Combine probabilities based on the weights
        log_prob = torch.log(self.weight1 * prob1 + self.weight2 * prob2)
        return log_prob