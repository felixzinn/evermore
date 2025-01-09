from __future__ import annotations

import pathlib

# import equinox as eqx
import jax
import jax.numpy as jnp
import law
import law.decorator
import luigi
from luigi.util import inherits, requires
from utils import (
    data_obs,
    fixed_mu_fit,
    generate_toys,
    histograms,
    model,
    parameters,
    plot_scan,
    q_mu,
    q_mu_vmap,
    # qmu_vmap_sampled_nuisances,
)

import evermore as evm

law.contrib.load("numpy")


class BaseParams(luigi.Config):
    n_toys = luigi.IntParameter(default=1_000, description="Number of toys to run")
    scan_range = law.CSVParameter(
        default=(0, 5, 0.1), description="Range of scan (jnp.arange)"
    )


@inherits(BaseParams)
class BaseTask(law.Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scan_range = tuple(float(val) for val in self.scan_range)

    @property
    def base_path(self) -> pathlib.Path:
        return pathlib.Path(__file__).parent.resolve()

    def local_target(self, *path) -> law.LocalFileTarget:
        return law.LocalFileTarget(
            self.base_path.joinpath("output", self.__class__.__name__, *path)
        )

    @property
    def store_parts(self):
        return (
            f"range_{'_'.join(map(str, self.scan_range))}",
            f"toys_{self.n_toys}",
        )


class LimitTask(BaseTask):
    """Scan over range of mu values and calculate CLs."""

    skip_output_removal = True

    def output(self) -> law.LocalFileTarget:
        return self.local_target(
            *self.store_parts,
            "scan.npy",
        )

    @law.decorator.timeit
    def run(self) -> None:
        # run limit scan and save to file
        key = jax.random.key(1)
        mu_vals = jnp.arange(*self.scan_range)

        p_values = []
        for step, mu_val in enumerate(mu_vals):
            if step % 10 == 0:
                print(f"{step=}")
            q_obs, *_ = q_mu(mu_val, parameters, histograms, data_obs)

            # STEP 4: find theta_obs(mu=0), theta_obs(mu=mu_test)
            parameters_bkgonly, _ = fixed_mu_fit(
                jnp.array([0]), parameters, histograms, data_obs
            )
            parameters_splusb, _ = fixed_mu_fit(
                mu_val, parameters, histograms, data_obs
            )

            # STEP 5: generate toy pseudo-data
            # from poisson
            # background only
            # expectation_bkgonly = evm.util.sum_over_leaves(
            #     model(parameters_bkgonly, histograms)
            # )
            # keys = jax.random.split(
            #     key, self.n_toys + 1
            # )  # one key for next key generation
            # toys_bkgonly = jax.vmap(evm.pdf.Poisson(expectation_bkgonly).sample)(
            #     keys[1:]
            # )
            # keys = jax.random.split(keys[0], self.n_toys + 1)
            # # random nuisance parameter value
            # params_sample_bkgonly = jax.vmap(evm.parameter.sample, in_axes=(None, 0))(
            #     parameters_bkgonly, keys[1:]
            # )
            # params_sample_bkgonly = eqx.tree_at(
            #     lambda tree: tree.mu.value,
            #     params_sample_bkgonly,
            #     jnp.zeros_like(toys_bkgonly),
            # )
            # q_mu_bkgonly = qmu_vmap_sampled_nuisances(
            #     parameters_bkgonly,
            #     histograms,
            #     (toys_bkgonly, params_sample_bkgonly),
            # )

            # # signal + background
            # expectation_splusb = evm.util.sum_over_leaves(
            #     model(parameters_splusb, histograms)
            # )
            # keys = jax.random.split(keys[0], self.n_toys + 1)
            # toys_splusb = jax.vmap(evm.pdf.Poisson(expectation_splusb).sample)(keys[1:])
            # keys = jax.random.split(keys[0], self.n_toys + 1)
            # # random nuisance parameter value
            # params_sample_splusb = jax.vmap(evm.parameter.sample, in_axes=(None, 0))(
            #     parameters_splusb, keys[1:]
            # )
            # params_sample_splusb = eqx.tree_at(
            #     lambda tree: tree.mu.value,
            #     params_sample_splusb,
            #     jnp.ones(self.n_toys) * mu_val,
            # )
            # q_mu_splusb = qmu_vmap_sampled_nuisances(
            #     parameters_bkgonly,
            #     histograms,
            #     (toys_splusb, params_sample_splusb),
            # )
            # key for next loop
            # key = keys[0]

            # generate toys with random nuisance values

            key, subkey = jax.random.split(key, 2)
            toys_bkgonly = evm.util.sum_over_leaves(
                generate_toys(
                    jnp.array([0]),
                    parameters_bkgonly,
                    histograms,
                    model,
                    self.n_toys,
                    subkey,
                )
            )
            key, subkey = jax.random.split(key, 2)
            toys_bkgonly = evm.pdf.Poisson(toys_bkgonly).sample(subkey)

            key, subkey = jax.random.split(key, 2)
            toys_splusb = evm.util.sum_over_leaves(
                generate_toys(
                    mu_val,
                    parameters_splusb,
                    histograms,
                    model,
                    self.n_toys,
                    key=subkey,
                ),
            )
            key, subkey = jax.random.split(subkey, 2)
            toys_splusb = evm.pdf.Poisson(toys_splusb).sample(subkey)

            # construct distribution of qmu for the hypotheses
            q_mu_bkgonly, *_ = q_mu_vmap(mu_val, parameters, histograms, toys_bkgonly)
            q_mu_splusb, *_ = q_mu_vmap(mu_val, parameters, histograms, toys_splusb)

            # STEP 6: calculate p_mu and pb
            p_mu = jnp.sum(q_mu_splusb >= q_obs) / self.n_toys
            pb = jnp.sum(q_mu_bkgonly >= q_obs) / self.n_toys

            # CLs = pmu / (1-pb)
            CLs = p_mu / pb
            p_values.append([p_mu, pb, CLs, mu_val])

        # save output
        self.output().parent.touch()
        jnp.save(self.output().path, jnp.array(p_values))


@requires(LimitTask)
class PlotLimitScan(BaseTask):
    # def requires(self):
    #     return LimitTask.req(self)

    def output(self):
        return self.local_target(*self.store_parts, "scan.png")

    def run(self):
        # load scan from file and plot

        inp = self.input().load()
        p_vals_splusb, p_vals_bkgonly, cls_vals, mu_vals = inp.T

        self.output().parent.touch()
        plot_scan(cls_vals, p_vals_splusb, p_vals_bkgonly, mu_vals, self.output().path)
