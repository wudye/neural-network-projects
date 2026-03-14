import numpy as np

def one_hot_numpy(labels, num_classes):
    return np.eye(num_classes)[labels]

def one_hot_manual(labels, num_classes):
    one_hot = np.zeros((len(labels),num_classes))
    for i, label in enumerate(labels):
        one_hot[i][label] = 1
    return one_hot

if __name__ == "__main__":
    labels = np.array([3, 5])
    num_classes = 10

    one_hot_encoded = one_hot_numpy(labels, num_classes)
    print(one_hot_encoded)
    one_hot_manual_encoded = one_hot_manual(labels, num_classes)
    print(one_hot_manual_encoded)