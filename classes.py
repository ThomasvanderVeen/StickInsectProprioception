from general import *

'''
This file holds the hair field, LIF and AdEx classes.
'''

class HairField:
    def __init__(self, parameters):
        self.N_hairs = parameters['N_hairs']
        self.max_joint_angle = parameters['max_joint_angle']
        self.max_list = parameters['max_joint_angle']
        self.min_joint_angle = parameters['min_joint_angle']
        self.min_list = parameters['min_joint_angle']
        self.max_angle = parameters['max_angle']
        self.overlap = parameters['overlap']
        self.receptive_field = None

    def get_receptive_field(self):
        """Calculate the receptive field edges based on the joint angle working range and overlap parameter."""
        rf = (self.max_joint_angle - self.min_joint_angle) / self.N_hairs
        rf *= (1 - self.overlap / (self.max_joint_angle - self.min_joint_angle))

        receptive_min = np.linspace(self.min_joint_angle,
                                    self.min_joint_angle + rf * (self.N_hairs - 1),
                                    num=self.N_hairs)

        receptive_max = np.linspace(self.max_joint_angle - rf * (self.N_hairs - 1),
                                    self.max_joint_angle,
                                    num=self.N_hairs)

        self.receptive_field = np.stack((receptive_min, receptive_max))

    def get_double_receptive_field(self):
        """Create a double receptive field by mirroring and concatenating the existing field."""
        self.get_receptive_field()
        rf_mirror = -self.receptive_field + (self.max_joint_angle + self.min_joint_angle)
        self.receptive_field = np.hstack((rf_mirror, self.receptive_field))
        self.N_hairs *= 2

    def reset_max_min(self, i):
        """Reset the max and min joint angles using the specified index."""
        self.max_joint_angle = self.max_list[i]
        self.min_joint_angle = self.min_list[i]

    def get_hair_angle(self, x):
        """Calculate the hair angle based on joint angle (x)."""
        min_rf = self.receptive_field[0, :]
        slope = self.max_angle / (self.receptive_field[1, :] - min_rf)

        slope_tiled = np.tile(slope, (x.size, 1))
        min_rf_tiled = np.tile(min_rf, (x.size, 1))
        x_tiled = np.tile(x, (self.N_hairs, 1)).T

        out = np.clip(slope_tiled * (x_tiled - min_rf_tiled), 0, 90)

        return out


class AdEx:
    NeuronState = namedtuple('NeuronState', ['V', 'w', 'spk'])

    def __init__(self, parameters):
        super(AdEx, self).__init__()
        self.C = parameters['C']
        self.g_L = parameters['g_L']
        self.E_L = parameters['E_L']
        self.DeltaT = parameters['DeltaT']
        self.a = parameters['a']
        self.V_T = parameters['V_T']
        self.tau_W = parameters['tau_W']
        self.b = parameters['b']
        self.V_R = parameters['V_R']
        self.V_cut = parameters['V_cut']
        self.n = parameters['n']
        self.dt = parameters['dt']
        self.state = None

    def initialize_state(self):
        """Initialize the neuron's state variables to None."""
        self.state = None

    def forward(self, input):
        """Perform a forward step of the AdEx model.

        Parameters:
        - input: The input current to the neuron.

        Returns:
        - V: Membrane potential after the update.
        - spk: Spike output of the neuron (binary indicator).
        """

        """If the self state is None, initialize the neurons initial values."""
        if self.state is None:
            self.state = self.NeuronState(V=np.linspace(self.E_L, self.E_L + 10e-3, self.n),
                                          w=np.zeros(self.n),
                                          spk=np.zeros(self.n))
        V = self.state.V
        w = self.state.w
        I = input

        V += (self.g_L * (self.E_L - V) + self.g_L * self.DeltaT * np.exp(
            (V - self.V_T) / self.DeltaT) - w + I) * self.dt / self.C

        spk = np.heaviside(V - self.V_cut, 0)

        V = (1 - spk) * V + spk * self.V_R

        w += spk * self.b
        w += (self.a * (V - self.E_L) - w) * self.dt / self.tau_W

        self.state = self.NeuronState(V=V, w=w, spk=spk)

        return V, spk


class LIF():
    NeuronState = namedtuple('NeuronState', ['V', 'w', 'spk', 'I'])

    def __init__(self, parameters):
        super(LIF, self).__init__()
        self.tau = parameters['tau']
        self.V_R = parameters['V_R']
        self.V_T = parameters['V_T']
        self.w = parameters['w']
        self.n = parameters['n']
        self.n_sims = parameters['n_sims']
        self.N_input = parameters['N_input']
        self.dt = parameters['dt']
        self.multiplesynapses = parameters['multiple_synapses']
        self.state = None

    def initialize_state(self):
        """Initialize the neuron's state variables to None."""
        self.state = None

    def forward(self, input):
        """Perform a forward step of the LIF model.

        Parameters:
        - input: The input spike to the neuron.

        Returns:
        - V: Membrane potential after the update.
        - spk: Spike output of the neuron (binary indicator).
        """

        """If the self state is None, initialize the neurons initial values."""
        if self.state is None:
            self.state = self.NeuronState(V=np.squeeze(np.full((self.n, self.n_sims), self.V_R)),
                                          w=np.squeeze(np.array(self.w)),
                                          spk=np.squeeze(np.zeros((self.n, self.n_sims))),
                                          I=np.squeeze(np.zeros((self.n, self.N_input, self.n_sims))))
        V = self.state.V
        w = self.state.w
        I = self.state.I

        V += self.dt*(self.V_R-V)/self.tau

        if self.multiplesynapses and self.n==1:
            V += np.sum(w * input, axis=0)
        elif self.multiplesynapses:
            V += np.sum(w * input, axis=1)
        else:
            V += w * input

        spk = np.heaviside(V - self.V_T, 0)
        V = (1 - spk) * V + spk * self.V_R

        self.state = self.NeuronState(V=V, w=w, spk=spk, I=I)



        return V, spk
