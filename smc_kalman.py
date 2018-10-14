import numpy as np

class LinearGaussianModel:
    def __init__(
        self,
        transition_model,
        transition_noise_covariance,
        observation_model,
        observation_noise_covariance,
        control_model = None):
        self.transition_model = transition_model
        self.transition_noise_covariance = transition_noise_covariance
        self.observation_model = observation_model
        self.observation_noise_covariance = observation_noise_covariance
        self.control_model = control_model

    def predict(
        self,
        state_mean_previous,
        state_covariance_previous,
        control_vector = None):
        state_mean = self.transition_model @ state_mean_previous + self.control_model @ control_vector
        state_covariance = self.transition_model @ state_covariance_previous @ self.transition_model.T + self.transition_noise_covariance
        return state_mean, state_covariance

    def observe(
        self,
        state_mean_prior,
        state_covariance_prior,
        observation_vector):
        kalman_gain_modified = state_covariance_prior @ self.observation_model.T @ np.linalg.inv(
            self.observation_model @ state_covariance_prior @ self.observation_model.T + self.observation_noise_covariance)
        state_mean_posterior = state_mean_prior + kalman_gain_modified @ (observation_vector - self.observation_model @ state_mean_prior)
        state_covariance_posterior = state_covariance_prior - kalman_gain_modified @ self.observation_model @ state_covariance_prior
        return state_mean_posterior, state_covariance_posterior

    def update(
        self,
        state_mean_previous,
        state_covariance_previous,
        observation_vector,
        control_vector = None):
        state_mean_prior, state_covariance_prior = self.predict(
            state_mean_previous,
            state_covariance_previous,
            control_vector)
        state_mean_posterior, state_covariance_posterior = self.observe(
            state_mean_prior,
            state_covariance_prior,
            observation_vector)
        return state_mean_posterior, state_covariance_posterior
