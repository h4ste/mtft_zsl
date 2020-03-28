def sentence_accuracy(references, predictions):
    """Compute accuracy, each line contains a label."""
    count = 0.0
    match = 0.0
    for label, pred in zip(references, predictions):
        if label == pred:
            match += 1
        count += 1
    return 100 * match / count


def word_accuracy(references, predictions):
    """Compute accuracy on per word basis."""
    total_acc, total_count = 0., 0.
    for labels, preds in zip(references, predictions):
        match = 0.0
        for pos in range(min(len(labels), len(preds))):
            label = labels[pos]
            pred = preds[pos]
            if label == pred:
                match += 1
        total_acc += 100 * match / max(len(labels), len(preds))
        total_count += 1
    return total_acc / total_count
