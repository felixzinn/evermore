from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt
import optax
from jaxtyping import Array

# from model import hists, model, observation
import evermore as evm

plt.rc("axes", grid=True)

hists = {
    "nominal": {
        "signal": jnp.array([10.0]),
        "bkg": jnp.array([50.0]),
    },
    "bkg_unc": jnp.array([7.0]),
}
observation = jnp.array([55.0])


def fun_bkg_unc(parameter: evm.Parameter, hist: Array) -> Array:
    return hist + parameter.value * hists["bkg_unc"]


class PyHFExample(eqx.Module):
    mu: evm.Parameter
    bkg_unc: evm.NormalParameter

    def __init__(self) -> None:
        evm.util.dataclass_auto_init(self)

    def __call__(self, hists: dict):
        expectations = {}
        # signal process
        sig_mod = self.mu.scale()
        expectations["signal"] = sig_mod(hists["nominal"]["signal"])

        # bkg process
        bkg_mod = evm.Modifier(
            parameter=self.bkg_unc,
            effect=evm.effect.Lambda(fun_bkg_unc, normalize_by="offset"),
        )
        expectations["bkg"] = bkg_mod(hists["nominal"]["bkg"])

        return expectations


model = PyHFExample()


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
    # apply parameter updates
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


def likelihood_ratio(
    nll_unconditional: Array, mu_val: float = 0.0
) -> tuple[float, Array]:
    """
    Likelihood ratio
    lambda (mu) = L(mu, theta_hat_hat) / L(mu_hat, theta_hat)

    return - 2 * ln(lambda ( 0 ))
    """
    nll_conditional, _ = calculate_nll_conditional(mu_val)
    # loss returns -ln(l), so the ratio is inverted

    return float(-2 * (nll_unconditional - nll_conditional))


def calculate_nll_unconditional() -> tuple[float, float]:
    model_unconditional, _ = unconditional_fit()
    return loss(
        *eqx.partition(
            model_unconditional, evm.parameter.value_filter_spec(model_unconditional)
        ),
        hists,
        observation,
    ), model_unconditional.mu.value


@jax.jit
def calculate_nll_conditional(mu_val: float) -> tuple[float, float]:
    model_conditional, _ = conditional_fit(mu_val=mu_val)
    return loss(
        *eqx.partition(
            model_conditional, evm.parameter.value_filter_spec(model_conditional)
        ),
        hists,
        observation,
    ), model_conditional.mu.value


def calculate_q0(mu_hat: float, likelihood_ratio: float) -> float:
    return jnp.where(mu_hat >= 0, likelihood_ratio, 0.0)


def calculate_qmu(mu_hat: float, mu: float, likelihood_ratio: float) -> float:
    return jnp.where(mu_hat <= mu, likelihood_ratio, 0.0)


"""
q0 test statistic
q0 = -2 * ln(lambda ( 0 )) if mu_hat >= 0
q0 = 0 if mu_hat < 0
"""
# get overall best fit and nll
nll_unconditional, mu_hat = calculate_nll_unconditional()
# likelihood ratio for best fit and mu=0
likelihood_ratio_value = likelihood_ratio(nll_unconditional=nll_unconditional)
q0 = calculate_q0(
    mu_hat, likelihood_ratio_value
)  # jnp.where(mu_hat >= 0, likelihood_ratio_value, 0.0)
p0 = 1 - jsp.stats.norm.cdf(jnp.sqrt(q0))

# https://github.com/scikit-hep/pyhf/blob/3d26434be836050e334190c212b1db1a6f7650c8/src/pyhf/infer/calculators.py#L170
p0_alt = jsp.stats.norm.cdf(-jnp.sqrt(q0))

# q_mu for upper limit
mu = 0.0
qmu = calculate_qmu(
    mu_hat, mu, likelihood_ratio_value
)  # jnp.where(mu_hat <= mu, likelihood_ratio_value, 0.0)
pmu = 1 - jsp.stats.norm.cdf(jnp.sqrt(qmu))

# CL_s+b
asimov_mu = 0.0
asimov_lr = likelihood_ratio(nll_unconditional=nll_unconditional, mu_val=asimov_mu)
q_asimov = calculate_qmu(mu_hat, asimov_mu, asimov_lr)
mu_vals = jnp.arange(0, 5, 0.1)
qmu_vals = []
pmu_vals = []
for mu in mu_vals:
    lr_val = likelihood_ratio(nll_unconditional=nll_unconditional, mu_val=mu)
    # qmu = jnp.where(mu_hat <= mu, lr_val, 0.0)
    # qmu = calculate_qmu(mu_hat, mu, lr_val)
    qmu = calculate_q0(mu_hat, lr_val)
    qmu_vals.append(qmu)
    pmu_vals.append(1 - jsp.stats.norm.cdf(jnp.sqrt(qmu)))

# CL_b, background only hypothesis
mu = 0.0
lr_val = likelihood_ratio(nll_unconditional=nll_unconditional, mu_val=mu)
# qb = jnp.where(mu_hat <= mu, lr_val, 0.0)
q_b = calculate_qmu(mu_hat, mu, lr_val)
p_b = 1 - jsp.stats.norm.cdf(jnp.sqrt(q_b))
CLs = [pmu / p_b for pmu in pmu_vals]


# plot results

fig, ax = plt.subplots()
ax.plot(mu_vals, qmu_vals, marker="o", ls="--")
ax.axvline(mu_hat, color="r", ls="--", label=r"$\hat{\mu}$")
ax.set_xlabel(r"$\mu$")
ax.set_ylabel(r"$q_\mu$")
ax.set_yscale("log")
ax.legend()

fig, ax = plt.subplots()
ax.plot(mu_vals, pmu_vals, marker="o", ls="--")
ax.axhline(0.05, color="k", ls=":", label=r"$\alpha = 0.05$")
ax.axvline(mu_hat, color="r", ls="--", label=r"$\hat{\mu}$")
ax.set_ylim([0, 1])
ax.set_xlabel(r"$\mu$")
ax.set_ylabel(r"$p_\mu$")
ax.legend()

fig, ax = plt.subplots()
ax.plot(mu_vals, CLs, marker="o", ls="--")
ax.axhline(0.05, color="k", ls=":", label=r"$\alpha = 0.05$")
ax.axvline(mu_hat, color="r", ls="--", label=r"$\hat{\mu}$")
ax.set_ylim([0, 1])
ax.set_xlabel(r"$\mu$")
ax.set_ylabel(r"$CL_s$")
ax.legend()
