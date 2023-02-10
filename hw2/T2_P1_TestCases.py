import numpy as np

'''
Instructions for using this Autograder:

0. In this problem, this autograder is used as a sanity check for your implementation of gradient descent.
1. Make sure it is in the same immediate directory/folder as your implementation file, which *must* be called T2_P1.py
2. Run this only after you have implemented basis2, basis3, and the LogisticRegressor class.
'''

def test_p1(LogisticRegressor, basis1, basis2, basis3):
    eta = 0.001
    runs = 10000
    
    # Input Data
    x = np.array([-8, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    y = np.array([1, 0, 1, 0, 0, 0, 1, 1, 1, 1]).reshape(-1, 1)
    
    TestWs = [np.array([[0.47662057],
       [0.10811676]]),np.array([[-0.32198979],
       [ 0.61626821],
       [ 0.1742742 ]]),np.array([[-0.52979913],
       [ 0.71183039],
       [-1.55714136],
       [10.86158108],
       [ 5.23025493],
       [ 0.47529788]])]
    
    modelTest = LogisticRegressor(eta=eta,runs=runs)

    x1 = basis1(x)
    if x1 is None:
        print("basis1 not implemented")
        return
    modelTest.fit(x1,y,w_init=np.ones((x1.shape[1], 1)))
    
    if len(TestWs[0])!=len(modelTest.W):
        print("Your w for basis1 has the wrong shape")
        return
    else:
        basis1_checker = np.allclose(TestWs[0], modelTest.W, rtol=0, atol=1e-2)
        if basis1_checker:
            basis1_checker = "Pass"
        else:
            basis1_checker = "Fail"
    
    x2 = basis2(x)
    if x2 is None:
        print("basis2 not implemented")
        return
    modelTest.fit(x2,y,w_init=np.ones((x2.shape[1], 1)))
    
    if len(TestWs[1])!=len(modelTest.W):
        print("Your w for basis2 has the wrong shape")
        return
    else:
        basis2_checker = np.allclose(TestWs[1], modelTest.W, rtol=0, atol=1e-2)
        if basis2_checker:
            basis2_checker = "Pass"
        else:
            basis2_checker = "Fail"
    
    x3 = basis3(x)
    if x3 is None:
        print("basis3 not implemented")
        return
    modelTest.fit(x3,y,w_init=np.ones((x3.shape[1], 1)))
    
    if len(TestWs[2])!=len(modelTest.W):
        print("Your w for basis3 has the wrong shape")
        return
    else:
        basis3_checker = np.allclose(TestWs[2], modelTest.W, rtol=0, atol=1e-1)
        if basis3_checker:
            basis3_checker = "Pass"
        else:
            basis3_checker = "Fail"
    
    print("Your test case results are, for basis 1, 2, and 3 respectively:", basis1_checker, basis2_checker, basis3_checker)
