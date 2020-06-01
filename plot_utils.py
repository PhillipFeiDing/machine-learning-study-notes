def plot_classifier_model(model, axis, cmap, grid=True):
    
    import numpy as np
    from collections import Counter
    import matplotlib.pyplot as plt
    
    x = np.arange(axis[0], axis[1], (axis[1] - axis[0]) / 100)
    y = np.arange(axis[2], axis[3], (axis[1] - axis[0]) / 75)
    (x_coor, y_coor) = np.meshgrid(x, y)
    
    X = np.hstack([x_coor.reshape(-1, 1), y_coor.reshape(-1, 1)])
    y_predict = model.predict(X)
    
    counter = Counter(y_predict)
    targets = sorted(list(counter.keys()))
    
    assert len(targets) <= len(cmap), "number of categories must match number of colors assigned"
    
    for i in range(0, len(targets)):
        (target, color) = (targets[i], cmap[i])
        plt.scatter(X[y_predict==target, 0], X[y_predict==target, 1], color=color, marker='o')
    
    return plt