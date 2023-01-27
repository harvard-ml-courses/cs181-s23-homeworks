import numpy as np
'''
Instructions for using this Autograder:
1. Make sure it is in the same immediate directory/folder as your implementation file, which *must* be called T1_P1.py
2. Run this only after you have implemented the function compute_loss, which returns the loss for a tau.
3. The test cases this Autograder uses are distinct from the tau values specified in the homework.
'''
def test_p1(kernel_regressor):
    tau1, y1 = 3, [ 6.15043922, -0.45956078, -0.40880182,  1.83049015, -4.29956078]
    tau2, y2 = 90, [ 5.80848422, -0.58094614,  0.41856207,  1.74314123, -3.73803587]
    tau3, y3 = 2700, [ 1.11280057, -1.13817445, -0.295227  , -0.97976764, -0.60896229]

    test_pts = np.array([400, 500, 600, 700, 800])
    train_data = np.genfromtxt("data/earth_temperature_sampled_train.csv", delimiter = ',')[1:]
    year_train = train_data[:, 0] / 1000
    temp_train = train_data[:, 1]

    for tau, y in zip([tau1, tau2, tau3], [y1, y2, y3]):
        assert np.allclose(y, kernel_regressor(test_pts, tau, year_train, temp_train)), f"Failed for tau={tau}"
    
    print("Passed")