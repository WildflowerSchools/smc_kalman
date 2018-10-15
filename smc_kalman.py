import numpy as np

class GaussianVector:
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

class LinearGaussianModel:
    def __init__(
        self,
        transition_model,
        transition_noise_covariance,
        observation_model,
        observation_noise_covariance,
        control_model = None):
        # Check properties of transition_model and coerce into desired format
        transition_model = np.asarray(transition_model)
        if transition_model.size == 1:
            transition_model = transition_model.reshape((1, 1))
        if transition_model.ndim != 2:
            raise ValueError('Specified transition model is not a two-dimensional matrix')
        if transition_model.shape[0] != transition_model.shape[1]:
            raise ValueError('Specified transition model is not a square matrix')
        self.transition_model = transition_model
        self.num_state_variables = self.transition_model.shape[0]
        # Check properties of transition_noise_covariance and coerce into desired format
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
        # Check properties of observation_model and coerce into desired format
        observation_model = np.asarray(observation_model)
        observation_model = observation_model.reshape(-1, self.num_state_variables)
        self.observation_model = observation_model
        self.num_observation_variables = self.observation_model.shape[0]
        # Check properties of observation_noise_covariance and coerce into desired format
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
        # Check properties of control_model and coerce into desired format
        if control_model is None:
            control_model = np.zeros((self.num_state_variables, 1))
        control_model = np.asarray(control_model)
        control_model = control_model.reshape((self.num_state_variables, -1))
        self.control_model = control_model
        self.num_control_variables = control_model.shape[1]

    def predict(
        self,
        previous_state,
        control_vector = None):
        # Check properties of previous_state_mean and coerce into desired format
        previous_state_mean = previous_state.mean
        if previous_state_mean.size != self.num_state_variables:
            raise ValueError('Size of previous state mean vector does not equal number of state variables in model')
        previous_state_mean = previous_state_mean.reshape(self.num_state_variables, 1)
        # Check properties of previous_state_covariance and coerce into desired format
        previous_state_covariance = previous_state.covariance
        if previous_state_covariance.size != self.num_state_variables**2:
            raise ValueError('Size of previous state covariance matrix does not equal number of state variables in model squared')
        previous_state_covariance = previous_state_covariance.reshape(self.num_state_variables, self.num_state_variables)
        # Check properties of control_vector and coerce into desired format
        if control_vector is None:
            control_vector = np.zeros((num_control_variables, 1))
        control_vector = np.asarray(control_vector)
        if control_vector.size != self.num_control_variables:
            raise ValueError('Size of control vector does not equal number of control variables in model')
        control_vector = control_vector.reshape(self.num_control_variables, 1)
        # Calculate the new state mean and covariance
        state_mean = self.transition_model @ previous_state_mean + self.control_model @ control_vector
        state_covariance = self.transition_model @ previous_state_covariance @ self.transition_model.T + self.transition_noise_covariance
        state_mean = np.squeeze(state_mean)
        return GaussianVector(state_mean, state_covariance)

    def observe(
        self,
        prior_state,
        observation_vector):
        # Check properties of prior_state_mean and coerce into desired format
        prior_state_mean = prior_state.mean
        if prior_state_mean.size != self.num_state_variables:
            raise ValueError('Size of prior state mean vector does not equal number of state variables in model')
        prior_state_mean = prior_state_mean.reshape(self.num_state_variables, 1)
        # Check properties of prior_state_covariance and coerce into desired format
        prior_state_covariance = prior_state.covariance
        if prior_state_covariance.size != self.num_state_variables**2:
            raise ValueError('Size of prior state covariance matrix does not equal number of state variables in model squared')
        prior_state_covariance = prior_state_covariance.reshape(self.num_state_variables, self.num_state_variables)
        # Check properties of observation_vector and coerce into desired format
        observation_vector = np.asarray(observation_vector)
        if observation_vector.size != self.num_observation_variables:
            raise ValueError('Size of observation vector does not equal number of observation variables in model')
        observation_vector = observation_vector.reshape(self.num_observation_variables, 1)
        # Calculate the posterior state mean and covariance
        kalman_gain_modified = prior_state_covariance @ self.observation_model.T @ np.linalg.inv(
            self.observation_model @ prior_state_covariance @ self.observation_model.T + self.observation_noise_covariance)
        posterior_state_mean = prior_state_mean + kalman_gain_modified @ (observation_vector - self.observation_model @ prior_state_mean)
        posterior_state_covariance = prior_state_covariance - kalman_gain_modified @ self.observation_model @ prior_state_covariance
        posterior_state_mean = np.squeeze(posterior_state_mean)
        return GaussianVector(posterior_state_mean, posterior_state_covariance)

    def update(
        self,
        previous_state,
        observation_vector,
        control_vector = None):
        prior_state = self.predict(
            previous_state,
            control_vector)
        posterior_state = self.observe(
            prior_state,
            observation_vector)
        return posterior_state
