import numpy as np


'''
Instructions for using this Autograder:
1. Make sure it is in the same immediate directory/folder as your implementation file, which *must* be called T1_P2.py
2. Run this only after you have implemented the functions predict_kernel and predict_knn.
'''

def test_p2(predict_knn):
    k1, y1 = 1, [ 6.15043922, -0.45956078, -0.40956078,  1.83043922, -4.29956078]
    k2, y2 = 3, [ 4.95377255, -0.60956078,  0.45043922,  1.94710589,  0.08043922]
    k3, y3 = 55, [-0.74010624, -0.83937896, -0.83937896, -0.83937896, -0.83937896]

    test_pts = np.array([400, 500, 600, 700, 800])
    train_data = np.genfromtxt("data/earth_temperature_sampled_train.csv", delimiter = ',')[1:]
    year_train = train_data[:, 0] / 1000
    temp_train = train_data[:, 1]

    for k, y in zip([k1, k2, k3], [y1, y2, y3]):
        assert np.allclose(y, predict_knn(test_pts, k, year_train, temp_train)), f"Failed for k={k}"

    print("Passed")