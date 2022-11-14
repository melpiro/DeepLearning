from numpy import array

def evaluate_output(true_values: array, prediction: array) -> float:
    total = true_values.shape[0]
    score = 0
    for i in range(total):
        if true_values[i] == prediction[i]:
            score += 1
    return score / total
    