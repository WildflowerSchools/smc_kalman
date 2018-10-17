import numpy as np

class GaussianDistribution:
    def __init__(
        self,
        mean,
        covariance):
        # Check properties of mean and coerce into desired format
        mean = np.asarray(mean)
        if np.squeeze(mean).ndim > 1:
            raise ValueError('Specified mean vector is not one-dimensional')
        mean = mean.reshape(mean.size)
        self.mean = mean
        self.num_variables = self.mean.shape[0]
        # Check properties of covariance and coerce into desired format
        covariance = np.asarray(covariance)
        if covariance.size == 1:
            covariance = covariance.reshape((1, 1))
        if covariance.ndim != 2:
            raise ValueError('Specified covariance is not a two-dimensional matrix')
        if covariance.shape[0] != covariance.shape[1]:
            raise ValueError('Specified covariance is not a square matrix')
        if covariance.shape[0] != self.num_variables:
            raise ValueError('Dimensions of specified covariance not equal to number of variables implied by mean vector')
        self.covariance = covariance

    def sample(
        self,
        num_samples = 1):
        samples = np.random.multivariate_normal(
            self.mean,
            self.covariance,
            num_samples)
        samples = np.squeeze(samples)
        return samples

class LinearGaussianModel:
    def __init__(
        self,
        transition_model,
        transition_noise_covariance,
        observation_model,
        observation_noise_covariance,
        control_model = None):
        # Check properties of transition model and coerce into desired format
        transition_model = np.asarray(transition_model)
        if transition_model.size == 1:
            transition_model = transition_model.reshape((1, 1))
        if transition_model.ndim != 2:
            raise ValueError('Specified transition model is not a two-dimensional matrix')
        if transition_model.shape[0] != transition_model.shape[1]:
            raise ValueError('Specified transition model is not a square matrix')
        self.transition_model = transition_model
        self.num_state_variables = self.transition_model.shape[0]
        # Check properties of transition noise covariance and coerce into desired format
        transition_noise_covariance = np.asarray(transition_noise_covariance)
        if transition_noise_covariance.size == 1:
            transition_noise_covariance = transition_noise_covariance.reshape((1, 1))
        if transition_noise_covariance.ndim != 2:
            raise ValueError('Specified transition noise covariance is not a two-dimensional matrix')
        if transition_noise_covariance.shape[0] != transition_noise_covariance.shape[1]:
            raise ValueError('Specified transition noise covariance is not a square matrix')
        if transition_noise_covariance.shape[0] != self.num_state_variables:
            raise ValueError('Dimensions of specified transition noise covariance not equal to number of state variables implied by specified transition model')
        self.transition_noise_covariance = transition_noise_covariance
        # Check properties of observation model and coerce into desired format
        observation_model = np.asarray(observation_model)
        observation_model = observation_model.reshape(-1, self.num_state_variables)
        self.observation_model = observation_model
        self.num_observation_variables = self.observation_model.shape[0]
        # Check properties of observation noise covariance and coerce into desired format
        observation_noise_covariance = np.asarray(observation_noise_covariance)
        if observation_noise_covariance.size == 1:
            observation_noise_covariance = observation_noise_covariance.reshape((1, 1))
        if observation_noise_covariance.ndim != 2:
            raise ValueError('Specified observation noise covariance is not a two-dimensional matrix')
        if observation_noise_covariance.shape[0] != observation_noise_covariance.shape[1]:
            raise ValueError('Specified observation noise covariance is not a square matrix')
        if observation_noise_covariance.shape[0] != self.num_observation_variables:
            raise ValueError('Dimensions of specified observation noise covariance not equal to number of observation variables implied by observation model')
        self.observation_noise_covariance = observation_noise_covariance
        # Check properties of control model and coerce into desired format
        if control_model is None:
            control_model = np.zeros((self.num_state_variables, 1))
        control_model = np.asarray(control_model)
        control_model = control_model.reshape((self.num_state_variables, -1))
        self.control_model = control_model
        self.num_control_variables = control_model.shape[1]

    def predict(
        self,
        previous_state_distribution,
        control_vector = None):
        # Check properties of previous state distribution mean and coerce into desired format
        previous_state_distribution_mean = previous_state_distribution.mean
        if previous_state_distribution_mean.size != self.num_state_variables:
            raise ValueError('Size of previous state distribution mean vector does not equal number of state variables in model')
        previous_state_distribution_mean = previous_state_distribution_mean.reshape(self.num_state_variables, 1)
        # Check properties of previous state distribution covariance and coerce into desired format
        previous_state_distribution_covariance = previous_state_distribution.covariance
        if previous_state_distribution_covariance.size != self.num_state_variables**2:
            raise ValueError('Size of previous state distribution covariance matrix does not equal number of state variables in model squared')
        previous_state_distribution_covariance = previous_state_distribution_covariance.reshape(self.num_state_variables, self.num_state_variables)
        # Check properties of control vector and coerce into desired format
        if control_vector is None:
            control_vector = np.zeros((num_control_variables, 1))
        control_vector = np.asarray(control_vector)
        if control_vector.size != self.num_control_variables:
            raise ValueError('Size of control vector does not equal number of control variables in model')
        control_vector = control_vector.reshape(self.num_control_variables, 1)
        # Calculate the current state distribution mean and covariance
        current_state_distribution_mean = self.transition_model @ previous_state_distribution_mean + self.control_model @ control_vector
        current_state_distribution_covariance = self.transition_model @ previous_state_distribution_covariance @ self.transition_model.T + self.transition_noise_covariance
        current_state_distribution_mean = np.squeeze(current_state_distribution_mean)
        return GaussianDistribution(current_state_distribution_mean, current_state_distribution_covariance)

    def observe(
        self,
        state_distribution):
        # Check properties of state distribution mean and coerce into desired format
        state_distribution_mean = state_distribution.mean
        if state_distribution_mean.size != self.num_state_variables:
            raise ValueError('Size of state distribution mean vector does not equal number of state variables in model')
        state_distribution_mean = state_distribution_mean.reshape(self.num_state_variables, 1)
        # Check properties of state distribution covariance and coerce into desired format
        state_distribution_covariance = state_distribution.covariance
        if state_distribution_covariance.size != self.num_state_variables**2:
            raise ValueError('Size of state distribution covariance matrix does not equal number of state variables in model squared')
        state_distribution_covariance = state_distribution_covariance.reshape(self.num_state_variables, self.num_state_variables)
        # Calculate the observation distribution mean and covariance
        observation_distribution_mean = self.observation_model @ state_distribution_mean
        observation_distribution_covariance = self.observation_model @ state_distribution_covariance @ self.observation_model.T + self.observation_noise_covariance
        observation_distribution_mean = np.squeeze(observation_distribution_mean)
        return GaussianDistribution(observation_distribution_mean, observation_distribution_covariance)

    def incorporate_observation(
        self,
        prior_state_distribution,
        observation_vector):
        # Check properties of prior state distribution mean and coerce into desired format
        prior_state_distribution_mean = prior_state_distribution.mean
        if prior_state_distribution_mean.size != self.num_state_variables:
            raise ValueError('Size of prior state distribution mean vector does not equal number of state variables in model')
        prior_state_distribution_mean = prior_state_distribution_mean.reshape(self.num_state_variables, 1)
        # Check properties of prior state distribution covariance and coerce into desired format
        prior_state_distribution_covariance = prior_state_distribution.covariance
        if prior_state_distribution_covariance.size != self.num_state_variables**2:
            raise ValueError('Size of prior state distribution covariance matrix does not equal number of state variables in model squared')
        prior_state_distribution_covariance = prior_state_distribution_covariance.reshape(self.num_state_variables, self.num_state_variables)
        # Check properties of observation vector and coerce into desired format
        observation_vector = np.asarray(observation_vector)
        if observation_vector.size != self.num_observation_variables:
            raise ValueError('Size of observation vector does not equal number of observation variables in model')
        observation_vector = observation_vector.reshape(self.num_observation_variables, 1)
        # Calculate the posterior state distribution mean and covariance
        kalman_gain_modified = prior_state_distribution_covariance @ self.observation_model.T @ np.linalg.inv(
            self.observation_model @ prior_state_distribution_covariance @ self.observation_model.T + self.observation_noise_covariance)
        posterior_state_distribution_mean = prior_state_distribution_mean + kalman_gain_modified @ (observation_vector - self.observation_model @ prior_state_distribution_mean)
        posterior_state_distribution_covariance = prior_state_distribution_covariance - kalman_gain_modified @ self.observation_model @ prior_state_distribution_covariance
        posterior_state_distribution_mean = np.squeeze(posterior_state_distribution_mean)
        return GaussianDistribution(posterior_state_distribution_mean, posterior_state_distribution_covariance)

    def update(
        self,
        previous_state_distribution,
        observation_vector,
        control_vector = None):
        current_state_distribution = self.predict(
            previous_state_distribution,
            control_vector)
        posterior_state_distribution = self.incorporate_observation(
            current_state_distribution,
            observation_vector)
        return posterior_state_distribution

    def simulate_prediction(
        self,
        previous_state,
        control_vector):
        # Check properties of previous state and coerce into desired format
        previous_state = np.asarray(previous_state)
        if np.squeeze(previous_state).ndim > 1:
            raise ValueError('Specified previous state vector is not one-dimensional')
        if previous_state.size != self.num_state_variables:
            raise ValueError('Size of previous state vector does not equal number of state variables in model')
        previous_state_distribution_mean = previous_state
        previous_state_distribution_covariance = np.zeros((self.num_state_variables, self.num_state_variables))
        previous_state_distribution = GaussianDistribution(
            previous_state_distribution_mean,
            previous_state_distribution_covariance)
        current_state_distribution = self.predict(
            previous_state_distribution,
            control_vector)
        current_state = current_state_distribution.sample()
        return current_state

    def simulate_observation(
        self,
        state):
        # Check properties of state and coerce into desired format
        state = np.asarray(state)
        if np.squeeze(state).ndim > 1:
            raise ValueError('Specified state vector is not one-dimensional')
        if state.size != self.num_state_variables:
            raise ValueError('Size of state vector does not equal number of state variables in model')
        state_distribution_mean = state
        state_distribution_covariance = np.zeros((self.num_state_variables, self.num_state_variables))
        state_distribution = GaussianDistribution(
            state_distribution_mean,
            state_distribution_covariance)
        observation_distribution = self.observe(
            state_distribution)
        observation = observation_distribution.sample()
        return observation
