import logging
import math
from typing import Any, Dict, Optional, Tuple

import torch
from sbi import inference as inference
from sbi.utils.get_nn_models import classifier_nn

from sbibm.algorithms.sbi.utils import (
    wrap_posterior,
    wrap_prior_dist,
    wrap_simulator_fn,
)
from sbibm.tasks.task import Task


def run(
    task: Task,
    num_samples: int,
    num_simulations: int,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    num_rounds: int = 10,
    neural_net: str = "resnet",
    hidden_features: int = 50,
    simulation_batch_size: int = 1000,
    training_batch_size: int = 10000,
    num_atoms: int = 10,
    automatic_transforms_enabled: bool = True,
    mcmc_method: str = "slice_np_vectorized",
    mcmc_parameters: Dict[str, Any] = {
        "num_chains": 100,
        "thin": 10,
        "warmup_steps": 25,
        "init_strategy": "sir",
        # NOTE: sir kwargs changed: num_candidate_samples = num_batches * batch_size
        "init_strategy_parameters": {
            "num_candidate_samples": 10000,
        },
    },
    z_score_x: str = "independent",
    z_score_theta: str = "independent",
    variant: str = "B",
    max_num_epochs: int = 2**31 - 1,
) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
    """Runs (S)NRE from `sbi`

    Args:
        task: Task instance
        num_samples: Number of samples to generate from posterior
        num_observation: Observation number to load, alternative to `observation`
        observation: Observation, alternative to `num_observation`
        num_simulations: Simulation budget
        num_rounds: Number of rounds
        neural_net: Neural network to use, one of linear / mlp / resnet
        hidden_features: Number of hidden features in network
        simulation_batch_size: Batch size for simulator
        training_batch_size: Batch size for training network
        num_atoms: Number of atoms, -1 means same as `training_batch_size`
        automatic_transforms_enabled: Whether to enable automatic transforms
        mcmc_method: MCMC method
        mcmc_parameters: MCMC parameters
        z_score_x: Whether to z-score x
        z_score_theta: Whether to z-score theta
        variant: Can be used to switch between SNRE-A (AALR) and -B (SRE)
        max_num_epochs: Maximum number of epochs

    Returns:
        Samples from posterior, number of simulator calls, log probability of true params if computable
    """
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)

    log = logging.getLogger(__name__)

    if num_rounds == 1:
        log.info(f"Running NRE")
        num_simulations_per_round = num_simulations
    else:
        log.info(f"Running SNRE")
        num_simulations_per_round = math.floor(num_simulations / num_rounds)

    if simulation_batch_size > num_simulations_per_round:
        simulation_batch_size = num_simulations_per_round
        log.warn("Reduced simulation_batch_size to num_simulation_per_round")

    if training_batch_size > num_simulations_per_round:
        training_batch_size = num_simulations_per_round
        log.warn("Reduced training_batch_size to num_simulation_per_round")

    prior = task.get_prior_dist()
    if observation is None:
        observation = task.get_observation(num_observation)

    simulator = task.get_simulator(max_calls=num_simulations)

    transforms = task._get_transforms(automatic_transforms_enabled)[
        "parameters"
    ]
    if automatic_transforms_enabled:
        prior = wrap_prior_dist(prior, transforms)
        simulator = wrap_simulator_fn(simulator, transforms)

    classifier = classifier_nn(
        model=neural_net.lower(),
        hidden_features=hidden_features,
        z_score_x=z_score_x,
        z_score_theta=z_score_theta,
    )
    if variant == "A":
        print("SNRE-A")
        inference_class = inference.SNRE_A
        training_kwargs = {}
    elif variant == "B":
        print("SNRE-B")
        inference_class = inference.SNRE_B
        training_kwargs = {"num_atoms": num_atoms}
    elif variant == "C":
        print("SNRE-C")
        inference_class = inference.SNRE_C
        training_kwargs = {"num_classes": 5,"gamma": 1.0}
    elif variant == "D":
        print("BNRE")
        inference_class = inference.BNRE
        training_kwargs = {"regularization_strength": 100.0}
    else:
        raise NotImplementedError

    inference_method = inference_class(classifier=classifier, prior=prior)

    posteriors = []
    proposal = prior

    for r in range(num_rounds):
        theta, x = inference.simulate_for_sbi(
            simulator,
            proposal,
            num_simulations=num_simulations_per_round,
            simulation_batch_size=simulation_batch_size,
        )

        ratio_estimator = inference_method.append_simulations(
            theta, x, from_round=r
        ).train(
            training_batch_size=training_batch_size,
            retrain_from_scratch=False,
            discard_prior_samples=False,
            show_train_summary=True,
            max_num_epochs=max_num_epochs,
            **training_kwargs,
        )

        (
            potential_fn,
            theta_transform,
        ) = inference.ratio_estimator_based_potential(
            ratio_estimator,
            prior,
            observation,
            # NOTE: disable transform if sbibm does it. will return IdentityTransform.
            enable_transform=not automatic_transforms_enabled,
        )
        posterior = inference.MCMCPosterior(
            potential_fn=potential_fn,
            proposal=prior,  # proposal for init_strategy
            theta_transform=theta_transform,
            method=mcmc_method,
            **mcmc_parameters,
        )
        # Change init_strategy to latest_sample after second round.
        if r > 1:
            posterior.init_strategy = "latest_sample"
            # copy init params from round 2 posterior.
            posterior._mcmc_init_params = posteriors[-1]._mcmc_init_params
        proposal = posterior.set_default_x(observation)
        posteriors.append(posterior)

    posterior = wrap_posterior(posteriors[-1], transforms)

    assert simulator.num_simulations == num_simulations

    samples = posterior.sample((num_samples,)).detach()

    return samples, simulator.num_simulations, None
