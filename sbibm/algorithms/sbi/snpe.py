import logging
import math
from typing import Optional, Tuple

import torch
from sbi import inference as inference
from sbi.utils.get_nn_models import posterior_nn

from sbibm.algorithms.sbi.utils import (
    wrap_posterior,
    wrap_prior_dist,
    wrap_simulator_fn,
)
from sbibm.tasks.task import Task
from sbibm.metrics import c2st, mmd
import csv

def run(
    task: Task,
    num_samples: int,
    num_simulations: int,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    num_rounds: int = 10,
    neural_net: str = "nsf",
    hidden_features: int = 50,
    simulation_batch_size: int = 1000,
    training_batch_size: int = 10000,
    num_atoms: int = 10,
    automatic_transforms_enabled: bool = False,
    z_score_x: str = "independent",
    z_score_theta: str = "independent",
    max_num_epochs: Optional[int] = 2**31 - 1,
    variant: str = "C",
) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
    """Runs (S)NPE from `sbi`

    Args:
        task: Task instance
        num_samples: Number of samples to generate from posterior
        num_simulations: Simulation budget
        num_observation: Observation number to load, alternative to `observation`
        observation: Observation, alternative to `num_observation`
        num_rounds: Number of rounds
        neural_net: Neural network to use, one of maf / mdn / made / nsf
        hidden_features: Number of hidden features in network
        simulation_batch_size: Batch size for simulator
        training_batch_size: Batch size for training network
        num_atoms: Number of atoms, -1 means same as `training_batch_size`
        automatic_transforms_enabled: Whether to enable automatic transforms
        z_score_x: Whether to z-score x
        z_score_theta: Whether to z-score theta
        max_num_epochs: Maximum number of epochs
        variant: Can be used to switch between SNPE-A and -C (APT)

    Returns:
        Samples from posterior, number of simulator calls, log probability of true params if computable
    """
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)

    log = logging.getLogger(__name__)

    if num_rounds == 1:
        log.info(f"Running NPE")
        num_simulations_per_round = num_simulations
    else:
        log.info(f"Running SNPE")
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

    transforms = task._get_transforms(automatic_transforms_enabled)["parameters"]
    
    if automatic_transforms_enabled:
        prior = wrap_prior_dist(prior, transforms)
        simulator = wrap_simulator_fn(simulator, transforms)

    density_estimator_fun = posterior_nn(
        model=neural_net.lower(),
        hidden_features=hidden_features,
        z_score_x=z_score_x,
        z_score_theta=z_score_theta,
    )
    if variant=="C":
        print("SNPE-C")
        inference_method = inference.SNPE_C(prior, density_estimator=density_estimator_fun)
        training_kwargs = {"num_atoms":num_atoms,"discard_prior_samples":False,
                "use_combined_loss":False}
    elif variant=="A":
        print("SNPE-A")
        #only the density estimator "mdn_snpe_a" works
        inference_method = inference.SNPE_A(prior,num_components=10)
        training_kwargs = {}
    elif variant=="B":
        print("SNPE-B")
        inference_method = inference.SNPE_B(prior=prior, density_estimator=neural_net, observation=observation)
        training_kwargs = {}
    elif variant=="D":
        print("SNPE-D")
        inference_method = inference.SNPE_D(prior=prior, density_estimator=neural_net)
        training_kwargs = {}
    else:
        raise NotImplementedError
    
    posteriors = []
    proposal = prior
    
    with open(f"snpe_{variant}_accuracy_{num_observation}_obs_{neural_net}.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "c2st", "mmd"])

    for r in range(num_rounds):
        print(f"round {r}")
        theta, x = inference.simulate_for_sbi(
            simulator,
            proposal,
            num_simulations=num_simulations_per_round,
            simulation_batch_size=simulation_batch_size,
        )
        
        density_estimator = inference_method.append_simulations(
            theta, x, proposal=proposal)
                
        density_estimator = density_estimator.train(
            training_batch_size=training_batch_size,
            retrain_from_scratch=False,
            show_train_summary=True,
            max_num_epochs=max_num_epochs,
            #force_first_round_loss=True,
            **training_kwargs,
        )
        posterior = inference_method.build_posterior(density_estimator)
        proposal = posterior.set_default_x(observation)
        
        posteriors.append(posterior)
        
        posterior_sampling = posterior.set_default_x(observation)
        posterior_sampling = wrap_posterior(posterior, transforms)
        posterior_samples = posterior_sampling.sample((num_samples,))
        reference_samples = task.get_reference_posterior_samples(num_observation=num_observation)
        
        accuracy_c2st = c2st(reference_samples, posterior_samples).item()
        accuracy_mmd = mmd(reference_samples, posterior_samples).item()
        
        with open(f"snpe_{variant}_accuracy_{num_observation}_obs_{neural_net}.csv",  "a") as f:
            writer = csv.writer(f)
            writer.writerow([r+1, accuracy_c2st, accuracy_mmd])

    posterior = wrap_posterior(posteriors[-1], transforms)
    
    assert simulator.num_simulations == num_simulations

    samples = posterior.sample((num_samples,)).detach()

    if num_observation is not None:
        true_parameters = task.get_true_parameters(num_observation=num_observation)
        log_prob_true_parameters = posterior.log_prob(true_parameters)
        return samples, simulator.num_simulations, log_prob_true_parameters
    else:
        return samples, simulator.num_simulations, None
