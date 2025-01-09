from __future__ import annotations

import os
from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jaxtyping import Array, PyTree
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import interp1d

import evermore as evm
from evermore.parameter import Parameter

jax.config.update("jax_enable_x64", True)

# %% Define the model
histograms = {
    "signal": jnp.array([10.0]),
    "background": jnp.array([50.0]),
    # "bkg_unc": jnp.array([7.0]),
}
data_obs = jnp.array([55.0])


# def fun_bkg_unc(parameter: evm.Parameter, hist: Array) -> Array:
#     return hist + parameter.value * histograms["bkg_unc"]


class Params(NamedTuple):
    mu: evm.Parameter
    bkg_unc: evm.NormalParameter


parameters = Params(
    mu=evm.Parameter(1.0, lower=-10.0, upper=10.0),
    bkg_unc=evm.NormalParameter(),
)


def model(params: Params, hists: dict[str, Array]) -> dict[str, Array]:
    expectation = {}

    # mod_background = evm.Modifier(
    #     parameter=params.bkg_unc,
    #     effect=evm.effect.Lambda(fun_bkg_unc, normalize_by="offset"),
    # )
    mod_background = params.bkg_unc.scale_log(up=1.1, down=0.9)
    mod_signal = params.mu.scale()  # @ mod_background

    expectation["signal"] = mod_signal(hists["signal"])
    expectation["background"] = mod_background(hists["background"])

    return expectation


# %% define the likelihood function
@eqx.filter_jit
def NLL(
    diff_params: PyTree,
    static_params: PyTree,
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
optim = optax.adam(learning_rate=1e-2)


@eqx.filter_jit
def make_step(
    params: PyTree,
    opt_state,
    hists: dict[str, Array],
    observation: Array,
    filter_spec,
) -> tuple:
    diff_params, static_params = eqx.partition(params, filter_spec)
    grads = eqx.filter_grad(NLL)(diff_params, static_params, hists, observation)
    updates, opt_state = optim.update(grads, opt_state)
    params = eqx.apply_updates(params, updates)
    return params, opt_state


def fit(
    params: PyTree,
    hists: dict[str, Array],
    observation: Array,
    steps: int = 5_000,
) -> tuple:
    """Minimize unconditional likelihood."""

    filter_spec = evm.parameter.value_filter_spec(params)

    def fun(step, params_optstate):
        params, opt_state = params_optstate
        return make_step(
            params,
            opt_state,
            hists,
            observation,
            filter_spec,
        )

    # needed for adam optimizer since carry input/output otherwise not the same
    opt_state = optim.init(
        eqx.filter(
            params,  # eqx.filter(params, eqx.is_inexact_array)
            filter_spec,
        )
    )

    return jax.lax.fori_loop(0, steps, fun, (params, opt_state))


@eqx.filter_jit
def fixed_mu_fit(
    mu_test: Array,
    params: PyTree[Parameter],
    hists: dict[str, Array],
    observation: Array,
    steps: int = 5_000,
) -> tuple:
    """Minimize likelihood for a fixed value of mu (conditional)."""

    filter_spec = evm.parameter.value_filter_spec(params)
    filter_spec = eqx.tree_at(
        lambda tree: tree.mu.value,
        filter_spec,
        replace=False,
    )

    def fun(step, params_optstate):
        params, opt_state = params_optstate

        return make_step(params, opt_state, hists, observation, filter_spec)

    # fix mu to value
    params_fixed_mu = eqx.tree_at(lambda t: t.mu.value, params, mu_test)

    opt_state = optim.init(eqx.filter(params, filter_spec))

    return jax.lax.fori_loop(0, steps, fun, (params_fixed_mu, opt_state))


# %% Construct the test statistic


def calculate_qmu(
    params_global: PyTree[Parameter],
    params_cond: PyTree[Parameter],
    hists: dict[str, Array],
    observation: Array,
) -> Array:
    mu_hat = params_global.mu.value
    # # enfore mu > 0
    # mu_hat_constraint = jnp.where(mu_hat < 0, 0, mu_hat)
    # params_global_constraint = eqx.tree_at(
    #     lambda tree: tree.mu.value, params_global, mu_hat_constraint
    # )

    # calculate unconditional likelihood
    nll_global = NLL(
        *eqx.partition(params_global, evm.parameter.value_filter_spec),
        hists,
        observation,
    )

    # calculate conditional likelihood
    nll_conditional = NLL(
        *eqx.partition(params_cond, evm.parameter.value_filter_spec),
        hists,
        observation,
    )
    mu_test = params_cond.mu.value

    return jnp.where(
        mu_hat > mu_test, jnp.array([0]), 2 * (nll_conditional - nll_global)
    )


sample_vmap = jax.vmap(evm.parameter.sample, in_axes=(None, 0))


def generate_toys(
    mu_test: Array,
    params: PyTree[Parameter],
    hists: dict[str, Array],
    toy_model: Callable,
    n_toys: int,
    key: jax.random.PRNGKey,
) -> Array:
    keys = jax.random.split(key, n_toys + 1)

    bkgunc_sample = sample_vmap(params.bkg_unc, keys[1:])
    # set bkg_unc values to sampled values
    params_sample = eqx.tree_at(
        lambda tree: tree.bkg_unc,
        params,
        bkgunc_sample,
    )
    # set mu values to test value with appropriate shape
    params_sample = eqx.tree_at(
        lambda tree: tree.mu.value,
        params_sample,
        jnp.full_like(bkgunc_sample.value, mu_test),
    )

    return toy_model(params_sample, hists)


def q_mu(
    mu_test: Array,
    params: PyTree[Parameter],
    hists: dict[str, Array],
    observation: Array,
) -> tuple[Array, PyTree[Parameter], PyTree[Parameter]]:
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


def calc_qmu_sampled_nuisances(params_global, hists, obs_paramscond) -> Array:
    observation, parameters_conditional = obs_paramscond
    return calculate_qmu(params_global, parameters_conditional, hists, observation)


qmu_vmap_sampled_nuisances = jax.vmap(
    calc_qmu_sampled_nuisances, in_axes=(None, None, 0)
)


q_mu_vmap = jax.vmap(q_mu, in_axes=(None, None, None, 0))

# %% plotting


def find_crossing(
    CLS_vals: Array, mu_vals: Array, alpha: float = 0.05
) -> tuple[Array, Array]:
    """Find the crossing of CLs with alpha.

    Parameters
    ----------
    CLS_vals : Array
        CLs values
    mu_vals : Array
        Mu values
    alpha : float, optional
        1 - Confidence Level (alpha=0.05 -> CLs=0.95), by default 0.05

    Returns
    -------
    tuple[Array, Array]
        mu value, CLs value of the crossing (interpolated)
    """
    cls_interp = interp1d(mu_vals, CLS_vals, kind="cubic")
    mu_vals_interp = jnp.linspace(mu_vals.min(), mu_vals.max(), 1000)
    cls_vals_interp = cls_interp(mu_vals_interp)
    index = np.where(np.diff(np.sign(cls_vals_interp - alpha)))[0]
    return mu_vals_interp[index], cls_vals_interp[index]


def plot_scan(
    CLs_vals: Array,
    p_vals_splusb: Array,
    p_vals_bkgonly: Array,
    mu_vals: Array,
    filename: os.PathLike,
    best_fit_mu: Array | float = None,
):
    # find crossing
    mu_crossing, cls_crossing = find_crossing(CLs_vals, mu_vals)

    # plotting
    fig, (ax_cl, ax_p) = plt.subplots(nrows=2, figsize=(8, 9), sharex=True)

    # plot CLs
    ax_cl.scatter(mu_vals, CLs_vals, color="black")
    ax_cl.axhline(0.05, ls="dashed", color="red", label=r"$\alpha = 0.05$")
    if len(mu_crossing) == 1:
        ax_cl.axvline(
            mu_crossing,
            ls="dashed",
            color="blue",
            label=rf"$\mu_{{\mathrm{{limit}}}} = {mu_crossing[0]:.4f}$",
        )
    ax_cl.set_ylabel("CLs")

    # plot p-values
    ax_p.plot(
        mu_vals,
        p_vals_splusb,
        label="p_mu",
        ls="dashed",
        marker="o",
    )
    ax_p.plot(
        mu_vals,
        p_vals_bkgonly,
        label="p_b",
        ls="dashed",
        marker="o",
    )
    ax_p.set_xlabel(r"$\mu$")
    ax_p.set_ylabel("p-value")

    # set ticks
    ax_cl.yaxis.set_major_locator(MultipleLocator(0.1))
    ax_cl.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax_p.xaxis.set_major_locator(MultipleLocator(0.5))
    ax_p.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax_p.yaxis.set_major_locator(MultipleLocator(0.1))
    ax_p.yaxis.set_minor_locator(MultipleLocator(0.05))

    # set grid
    ax_cl.grid(which="major", ls="dotted", color="gray")
    ax_p.grid(which="major", ls="dotted", color="gray")

    # add vertical line at best_fit value
    if best_fit_mu is not None:
        ax_cl.axvline(best_fit_mu, ls="dashed", color="orange", label="global best fit")
        ax_p.axvline(best_fit_mu, ls="dashed", color="orange", label="global best fit")

    # legend
    ax_cl.legend()
    ax_p.legend()

    # save
    fig.tight_layout()
    fig.savefig(filename)
