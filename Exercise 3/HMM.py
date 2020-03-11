import numpy as np

# -----------index 0 is False, index 1 is True---------------
# row is Rain_t-1, col is Rain_t
TRANSITION_MODEL = np.array([[0.7, 0.3],
                             [0.3, 0.7]])

INIT_PROB = np.array([[0.5],
                      [0.5]])

# row is Rain_t, col is Umbrella_t
OBSERVATION_MODEL = np.array([[0.8, 0.2],
                              [0.1, 0.9]])
#-------------------------------------------------------------

def normalize(forward_messages):
    return forward_messages / np.sum(forward_messages, axis=0)

def forward(observations):
    forward_messages = np.hstack((INIT_PROB, np.zeros((len(TRANSITION_MODEL), len(observations)))))
    for t,e in enumerate(observations):
        forward_messages[:, t + 1] = OBSERVATION_MODEL[:, e] * np.dot(np.transpose(TRANSITION_MODEL), forward_messages[:, t])
    return forward_messages

def backward(observations):
    backward_messages = np.hstack((np.zeros((len(TRANSITION_MODEL), len(observations))), np.ones((len(TRANSITION_MODEL), 1))))
    for t in range(len(observations), 0, -1):
        k = t - 1
        backward_messages[:, k] = np.dot(np.transpose(OBSERVATION_MODEL[:, observations[k]]) * TRANSITION_MODEL, backward_messages[:, t])
    return backward_messages

def forward_backwards(observations):
    forward_messages = forward(observations)
    backward_messages = backward(observations)
    smoothed = forward_messages * backward_messages
    return normalize((smoothed))


print("--------------PART B---------------" + '\n')
# part B
observations1 = np.array([1, 1]) # we saw umbrella on t = 1 and t = 2
observations2 = np.array([1, 1, 0, 1, 1]) # 1 represents we saw umbrella at timestep t, 0 represents no umbrella

state = 1 # Rain at time T
forward_messages = normalize(forward(observations1))
print("* Probability of rain at day 2: " + str(forward_messages[state, len(observations1)]) + '\n')

forward_messages = normalize(forward(observations2))
print("* Probability of rain on day 5: " + str(forward_messages[state, len(observations2)]) + '\n')
print("Documentation:")
for t,e in enumerate(observations2):
    print("Normalized forward message at timestep " + str(t + 1) + ": " + str(forward_messages[state, t + 1]))
print('\n')

print("--------------PART C---------------" + '\n')

# Part C
probability = forward_backwards(observations1)[:, 1]
print("* Probability distribution of states at day 1: " + str(probability) + '\n') # this will print the array in opposite order as the exercise, because of the definition of index 0 representing false and 1 representing true

probability = forward_backwards(observations2)[state, 1]
print("* Probability of rain at day 1: " + str(probability) + '\n')
backwards_messages = backward(observations2)
print("Documentation:")
for t,e in enumerate(observations2):
    print("Normalized backward message at timestep " + str(t + 1) + ": " + str(backwards_messages[state, t + 1]))

