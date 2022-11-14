from numpy import array

def evaluate_output(labels: list, real: array, pred: array) -> None:
    total = len(list(zip(real, pred)))
    scores = dict([(key, 0) for key in labels])
        
    for idx, (x, y) in enumerate(zip(real, pred)):
        local_pred = 1 if y > 0.5 else 0
        print(idx)
        if x == local_pred:
            scores[labels[idx]] += 1
    
    print(scores)