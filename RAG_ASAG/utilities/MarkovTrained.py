import random
from collections import defaultdict


def train_markov_model(text, order=1):
    """
    Trains a Markov chain model by mapping 'states'
    to a list of possible next words.
    """
    words = text.split()
    model = defaultdict(list)

    # Iterate through the text to build the transition 'kernel'
    for i in range(len(words) - order):
        state = tuple(words[i: i + order])
        next_word = words[i + order]
        model[state].append(next_word)

    return model


def generate_text(model, start_state, length=10):
    """Generates text by randomly sampling from the trained transitions."""
    current_state = start_state
    result = list(start_state)

    for _ in range(length):
        if current_state not in model:
            break
        # Pick the next word based on observed frequencies
        next_word = random.choice(model[current_state])
        result.append(next_word)
        current_state = tuple(result[-len(start_state):])

    return " ".join(result)


# 1. Training Data (Small corpus)
data = "the cat sat on the mat the cat ate the fish the dog barked at the cat"

# 2. Train the model (Order 1 = looks at 1 previous word)
markov_chain = train_markov_model(data, order=1)

# 3. Generate a sequence
print(generate_text(markov_chain, ("the",), length=5))