"""Calculate Limits with Toys"""
# https://inspirehep.net/literature/1196797
# following the data of the one bin example in the README.md from pyhf

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jaxtyping import Array

import evermore as evm

jax.config.update("jax_enable_x64", True)

# %% setup the model

histograms = {
    "signal": jnp.array([10.0]),
    "background": jnp.array([50.0]),
    "bkg_unc": jnp.array([7.0]),
}
data_obs = jnp.array([55.0])


def fun_bkg_unc(parameter: evm.Parameter, hist: Array) -> Array:
    return hist + parameter.value * histograms["bkg_unc"]


class Params(NamedTuple):
    mu: evm.Parameter
    bkg_unc: evm.NormalParameter


parameters = Params(
    mu=evm.Parameter(1.0, lower=jnp.array(-10.0), upper=jnp.array(10.0)),
    bkg_unc=evm.NormalParameter(),
)


def model(params: Params, hists: dict[str, Array]) -> dict[str, Array]:
    expectation = {}

    mod_signal = params.mu.scale()
    mod_background = evm.Modifier(
        parameter=params.bkg_unc,
        effect=evm.effect.Lambda(fun_bkg_unc, normalize_by="offset"),
    )

    expectation["signal"] = mod_signal(hists["signal"])
    expectation["background"] = mod_background(hists["background"])

    return expectation


# %% STEP 1: Construct the likelihood function
@eqx.filter_jit
def NLL(
    diff_params: Params,
    static_params: Params,
    hists: dict[str, Array],
    observation: Array,
) -> Array:
    """Returns the negative log-likelihood value"""
    params = eqx.combine(diff_params, static_params)
    expectation = evm.util.sum_over_leaves(model(params, hists))

    # Poisson likelihood
    log_likelihood = evm.pdf.Poisson(expectation).log_prob(observation)

    # constraints
    constraints = evm.loss.get_log_probs(params)
    log_likelihood += evm.util.sum_over_leaves(constraints)
    return -jnp.sum(log_likelihood)


# define helper functions for minimization
optim = optax.sgd(learning_rate=1e-2)


@eqx.filter_jit
def make_step(
    params: Params, opt_state, hists: dict[str, Array], observation: Array, filter_spec
) -> tuple:
    diff_params, static_params = eqx.partition(params, filter_spec)
    grads = eqx.filter_grad(NLL)(diff_params, static_params, hists, observation)
    updates, opt_state = optim.update(grads, opt_state)
    params = eqx.apply_updates(params, updates)
    return params, opt_state


def fit(
    params: Params, hists: dict[str, Array], observation: Array, steps: int = 1000
) -> tuple:
    """Minimize unconditional likelihood."""

    def fun(step, params_optstate):
        params, opt_state = params_optstate
        return make_step(
            params,
            opt_state,
            hists,
            observation,
            evm.parameter.value_filter_spec(params),
        )

    opt_state = optim.init(eqx.filter(params, eqx.is_inexact_array))

    return jax.lax.fori_loop(0, steps, fun, (params, opt_state))


@eqx.filter_jit
def fixed_mu_fit(
    mu_test: Array,
    params: Params,
    hists: dict[str, Array],
    observation: Array,
    steps: int = 1000,
) -> tuple:
    """Minimize likelihood for a fixed value of mu (conditional)."""

    def fun(step, params_optstate):
        params, opt_state = params_optstate

        filter_spec = evm.parameter.value_filter_spec(params)
        filter_spec = eqx.tree_at(
            lambda tree: tree.mu.value,
            filter_spec,
            replace=False,
        )

        return make_step(params, opt_state, hists, observation, filter_spec)

    # fix mu to value
    params_fixed_mu = eqx.tree_at(lambda t: t.mu.value, params, mu_test)

    opt_state = optim.init(eqx.filter(params, eqx.is_inexact_array))

    return jax.lax.fori_loop(0, steps, fun, (params_fixed_mu, opt_state))


# %% STEP 2: Construct the test statistic


def calculate_qmu(
    params_global: Params,
    params_cond: Params,
    hists: dict[str, Array],
    observation: Array,
) -> Array:
    nll_global = NLL(
        *eqx.partition(params_global, evm.parameter.value_filter_spec),
        hists,
        observation,
    )
    mu_hat = params_global.mu.value

    nll_conditional = NLL(
        *eqx.partition(params_cond, evm.parameter.value_filter_spec),
        hists,
        observation,
    )
    mu_test = params_cond.mu.value

    return jnp.where(
        mu_hat > mu_test, jnp.array([0]), 2 * (nll_conditional - nll_global)
    )


def q_mu(
    mu_test: Array, params: Params, hists: dict[str, Array], observation: Array
) -> tuple[Array, Params, Params]:
    """Returns the test statistic q_mu."""
    # get global minimum and the corresponding likelihood value
    params_global, _ = fit(params, hists, observation)

    # get minimum for test value of mu
    params_conditional, _ = fixed_mu_fit(mu_test, params, hists, observation)

    return (
        calculate_qmu(params_global, params_conditional, hists, observation),
        params_global,
        params_conditional,
    )


q_mu_vmap = jax.vmap(q_mu, in_axes=(None, None, None, 0))


# %% STEP 3: find observed value of test statistic for given mu under test

mu_val = jnp.array([1.0])
q_obs, parameters_global, parameters_conditional = q_mu(
    mu_val, parameters, histograms, data_obs
)


# %% STEP 4: find theta_obs(mu=0), theta_obs(mu=mu_test)

# find the values of the nuisance parameters for mu=0 and mu=mu_test
# which best describe the experimentally observed data,
# i.e. that maximize the likelihood for the background only and signal+background hypotheses
parameters_bkgonly, _ = fixed_mu_fit(jnp.array([0]), parameters, histograms, data_obs)
parameters_splusb, _ = fixed_mu_fit(mu_val, parameters, histograms, data_obs)

# %% STEP 5: generate toy pseudo-data
# assuming signal strength mu in S+B, mu=0 in background only
# fix nuisance parameters for toy generation
# sample from poisson distribution

key = jax.random.key(0)
N_TOYS = 10_000

# background only
expectation_bkgonly = evm.util.sum_over_leaves(model(parameters_bkgonly, histograms))
keys = jax.random.split(key, N_TOYS + 1)  # one key for next key generation
toys_bkgonly = jax.vmap(evm.pdf.Poisson(expectation_bkgonly).sample)(keys[1:])

# signal + background
expectation_splusb = evm.util.sum_over_leaves(model(parameters_splusb, histograms))
keys = jax.random.split(keys[0], N_TOYS + 1)
toys_splusb = jax.vmap(evm.pdf.Poisson(expectation_splusb).sample)(keys[1:])

# construct distribution of qmu for the hypotheses
q_mu_bkgonly, *_ = q_mu_vmap(mu_val, parameters, histograms, toys_bkgonly)
q_mu_splusb, *_ = q_mu_vmap(mu_val, parameters, histograms, toys_splusb)

fig, ax = plt.subplots(figsize=(8, 4.5))

counts, bins, _ = ax.hist(
    q_mu_bkgonly, bins=jnp.arange(0, 10, 0.2), label="Bkg only", histtype="step"
)
ax.hist(q_mu_splusb, bins=bins, label="S+B", histtype="step")
ax.axvline(q_obs, ls="dotted", color="black", label=r"$q_{obs}$")

ax.set_yscale("log")
ax.set_xlabel(r"$q_\mu$")
ax.set_ylabel("Occurence")
ax.legend()
ax.set_title(rf"$\mu = {mu_val.item()}")

fig.tight_layout()
fig.savefig("limits_with_toys.png")

# %% STEP 6: calculate p_mu and pb
# p_mu = P(qmu >= qobs | S+B)
p_mu = jnp.sum(q_mu_splusb >= q_obs) / N_TOYS

# 1 - pb = P(qmu >= qobs | bkg-only)
pb = jnp.sum(q_mu_bkgonly >= q_obs) / N_TOYS

# %% STEP 7: CLs <= alpha -> exclude mu with (1-alpha) Confidence level

# CLs = pmu / (1-pb)
CLs = p_mu / pb
print(f"mu = {mu_val} => CLs = {CLs:.4f}")


# %% STEP 8: find CLs == 0.05 for 95 % CL

key = jax.random.key(0)
N_TOYS = 10_000

mu_vals = jnp.arange(0, 5, 0.1)
p_values = []
for mu_val in mu_vals:
    q_obs, parameters_global, parameters_conditional = q_mu(
        mu_val, parameters, histograms, data_obs
    )

    # STEP 4: find theta_obs(mu=0), theta_obs(mu=mu_test)
    parameters_bkgonly, _ = fixed_mu_fit(
        jnp.array([0]), parameters, histograms, data_obs
    )
    parameters_splusb, _ = fixed_mu_fit(mu_val, parameters, histograms, data_obs)

    # STEP 5: generate toy pseudo-data
    # background only
    expectation_bkgonly = evm.util.sum_over_leaves(
        model(parameters_bkgonly, histograms)
    )
    keys = jax.random.split(key, N_TOYS + 1)  # one key for next key generation
    toys_bkgonly = jax.vmap(evm.pdf.Poisson(expectation_bkgonly).sample)(keys[1:])

    # signal + background
    expectation_splusb = evm.util.sum_over_leaves(model(parameters_splusb, histograms))
    keys = jax.random.split(keys[0], N_TOYS + 1)
    toys_splusb = jax.vmap(evm.pdf.Poisson(expectation_splusb).sample)(keys[1:])

    # key for next loop
    key = keys[0]

    # construct distribution of qmu for the hypotheses
    q_mu_bkgonly, *_ = q_mu_vmap(mu_val, parameters, histograms, toys_bkgonly)
    q_mu_splusb, *_ = q_mu_vmap(mu_val, parameters, histograms, toys_splusb)

    # STEP 6: calculate p_mu and pb
    p_mu = jnp.sum(q_mu_splusb >= q_obs) / N_TOYS
    pb = jnp.sum(q_mu_bkgonly >= q_obs) / N_TOYS

    # CLs = pmu / (1-pb)
    CLs = p_mu / pb
    p_values.append([p_mu, pb, CLs])

# plot CLs scan
CLs_vals = [cls[-1].item() for cls in p_values]

fig, ax = plt.subplots(figsize=(8, 4.5))

ax.scatter(mu_vals, CLs_vals, color="black")
ax.plot(mu_vals, CLs_vals, ls="dotted", color="gray")
ax.set_xlabel(r"$mu$")
ax.set_ylabel("CLs")

fig.savefig("CLs_scan.png")
