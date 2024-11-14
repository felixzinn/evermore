from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import optax
from jaxtyping import Array
from model import hists, model, observation

import evermore as evm

# optimizer
optim = optax.sgd(learning_rate=1e-2)
opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))


@eqx.filter_jit
def loss(dynamic_model, static_model, hists, observation):
    model = eqx.combine(dynamic_model, static_model)
    expectations = model(hists)
    constraints = evm.loss.get_log_probs(model)
    loss_val = evm.pdf.Poisson(evm.util.sum_over_leaves(expectations)).log_prob(
        observation
    )
    # add constraint
    loss_val += evm.util.sum_over_leaves(constraints)
    return -jnp.sum(loss_val)


@eqx.filter_jit
def make_step(model, opt_state, events, observation, filter_spec):
    # differentiate full analysis
    dynamic_model, static_model = eqx.partition(model, filter_spec)
    grads = eqx.filter_grad(loss)(dynamic_model, static_model, events, observation)
    updates, opt_state = optim.update(grads, opt_state)
    # apply nuisance parameter and DNN weight updates
    model = eqx.apply_updates(model, updates)
    return model, opt_state


@jax.jit
def unconditional_fit(steps: int = 10_000) -> tuple[eqx.Module, tuple]:
    def fit(step, model_optstate):
        model, opt_state = model_optstate
        return make_step(
            model, opt_state, hists, observation, evm.parameter.value_filter_spec(model)
        )

    return jax.lax.fori_loop(0, steps, fit, (model, opt_state))


@jax.jit
def conditional_fit(mu_val: float, steps: int = 10_000) -> tuple[eqx.Module, tuple]:
    def fit(step, model_optstate):
        model, opt_state = model_optstate

        filter_spec = evm.parameter.value_filter_spec(model)
        filter_spec = eqx.tree_at(
            lambda tree: tree.mu.value,
            filter_spec,
            replace=False,
        )

        return make_step(model, opt_state, hists, observation, filter_spec)

    # fix mu to value
    model_fixed_mu = eqx.tree_at(lambda t: t.mu.value, model, jnp.array(mu_val))

    return jax.lax.fori_loop(0, steps, fit, (model_fixed_mu, opt_state))


def likelihood_ratio(mu_val: float = 0.0) -> tuple[float, Array]:
    """
    Likelihood ratio
    lambda (mu) = L(mu, theta_hat_hat) / L(mu_hat, theta_hat)

    return - 2 * ln(lambda ( 0 ))
    """
    model_unconditional, _ = unconditional_fit()
    model_conditional, _ = conditional_fit(mu_val=mu_val)

    nll_conditional = loss(
        *eqx.partition(
            model_conditional, evm.parameter.value_filter_spec(model_conditional)
        ),
        hists,
        observation,
    )
    nll_unconditional = loss(
        *eqx.partition(
            model_unconditional, evm.parameter.value_filter_spec(model_unconditional)
        ),
        hists,
        observation,
    )

    # loss returns -ln(l), so the ratio is inverted

    return float(
        -2 * (nll_unconditional - nll_conditional)
    ), model_unconditional.mu.value


"""
q0 test statistic
q0 = -2 * ln(lambda ( 0 )) if mu_hat >= 0
q0 = 0 if mu_hat < 0
"""
likelihood_ratio_value, mu_hat = likelihood_ratio()
q0 = jnp.where(mu_hat >= 0, likelihood_ratio_value, 0.0)
p0 = 1 - jsp.stats.norm.cdf(jnp.sqrt(q0))

# https://github.com/scikit-hep/pyhf/blob/3d26434be836050e334190c212b1db1a6f7650c8/src/pyhf/infer/calculators.py#L170
p0_alt = jsp.stats.norm.cdf(-jnp.sqrt(q0))

# q_mu for upper limit
mu = 0.0
qmu = jnp.where(mu_hat <= mu, likelihood_ratio_value, 0.0)
pmu = 1 - jsp.stats.norm.cdf(jnp.sqrt(qmu))

mu_vals = jnp.arange(0, 10, 0.5)
pmu_vals = []
for mu in mu_vals:
    lr_val, mu_hat = likelihood_ratio(mu)
    qmu = jnp.where(mu_hat <= mu, lr_val, 0.0)
    pmu_vals.append(1 - jsp.stats.norm.cdf(jnp.sqrt(qmu)))
