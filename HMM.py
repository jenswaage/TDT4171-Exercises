import numpy as np

# -----------index 0 is False, index 1 is True---------------
# row is Rain_t-1, col is Rain_t
TRANSITION_MODEL = np.array([[0.7, 0.3],
                             [0.3, 0.7]])
INIT_PROB = np.array([0.5, 0.5])
# row is Rain_t, col is Umbrella_t
OBSERVATION_MODEL = np.array([[0.8, 0.2],
                              [0.1, 0.9]])
#-------------------------------------------------------------

def forward(observations):
    probability_matrix = np.zeros((len(TRANSITION_MODEL), len(observations)))

    for t,e in enumerate(observations):
        for 




#----------- [F  T]


observations = np.array([1, 1]) # we saw umbrella on t = 1 and t = 2
forward(observations)
