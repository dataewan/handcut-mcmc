from scipy import stats
import numpy as np
import tqdm


def sampler(
    data, samples=4, mu_init=0.5, proposal_width=0.5, mu_prior_mu=0, mu_prior_sd=1
):
    mu_current = mu_init
    posterior = [mu_current]

    for i in tqdm.tqdm(range(samples)):
        mu_proposal = get_proposal(mu_current, proposal_width)

        likelihood_current, likelihood_proposal = compute_likelihoods(
            data, mu_current, mu_proposal
        )

        prior_current, prior_proposal = compute_priors(
            mu_prior_mu, mu_prior_sd, mu_current, mu_proposal
        )

        if should_accept(
            likelihood_current, likelihood_proposal, prior_current, prior_proposal
        ):
            mu_current = mu_proposal

        posterior.append(mu_current)

    return posterior


def get_proposal(mu_current, proposal_width):
    """Draw a new value from a normal distribution with mean of μ_current, and
    standard deviation of proposal_width."""
    return stats.norm(mu_current, proposal_width).rvs()


def compute_likelihoods(data, mu_current, mu_proposal):
    """Now evaluate if that's a good place to jump to or not. Trying to find
    out if the new μ is a better explanation of the data or not. We compute the
    probability of the data, given the likelihood (normal distribution) with
    the proposed values (μ = new value, standard deviation fixed as one).
    """
    likelihood_current = stats.norm(mu_current, 1).pdf(data).prod()
    likelihood_proposal = stats.norm(mu_proposal, 1).pdf(data).prod()
    return likelihood_current, likelihood_proposal


def compute_priors(mu_prior_mu, mu_prior_sd, mu_current, mu_proposal):
    """Computing the prior probabilities for the current and proposed μ values.
    """
    prior_current = stats.norm(mu_prior_mu, mu_prior_sd).pdf(mu_current)
    prior_proposal = stats.norm(mu_prior_mu, mu_prior_sd).pdf(mu_proposal)
    return prior_current, prior_proposal


def should_accept(
    likelihood_current, likelihood_proposal, prior_current, prior_proposal
):
    """Accept based on the ratio of the numerators of Bayes law for the new point and the old point.
    If our P for the new point is larger than the old one then we have to
    accept it. If the new P is smaller than the old one, there is still a
    chance that we will accept it.
    """
    p_current = likelihood_current * prior_current
    p_proposal = likelihood_proposal * prior_proposal

    p_accept = p_proposal / p_current

    return np.random.rand() < p_accept


if __name__ == "__main__":
    data = np.random.randn(20)
    posterior = sampler(data, samples=4, mu_init=1.)
