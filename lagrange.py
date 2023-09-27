import scipy.interpolate as sc
import numpy as np

test_x_points = [2.24360357, 2.9121162,  2.20650044, 2.67055196, 0.30076948, 1.88485279,
 0.05352137, 2.39245231, 0.85236733, 2.92144751, 1.95105661, 1.36673932,
 0.77551562, 2.1114804,  1.67575975, 1.82675308, 1.44179107, 0.08493861,
 2.4959921,  1.58823485, 0.71335608, 3.00355238, 2.56112735, 0.2445052,
 2.51345024, 0.44474771, 2.35316575, 0.50419046, 2.597295,   0.60388625,
 0.01299012, 0.7820196,  0.38576213, 1.78946331, 1.39085728, 1.89386325,
 1.34606327, 0.21259385, 1.05156696, 2.23738403, 1.25133785, 2.17713909,
 0.18395914, 2.73929532, 1.32091287, 0.09286735, 0.17511292, 0.57919145,
 2.89188621, 3.0099907,  2.39682123, 0.74373372, 2.81669713, 1.09293894,
 1.86361765, 0.55507441, 2.21278297, 0.17061377, 2.7057526,  2.92664688,
 1.33984509, 1.05562774, 1.40302981, 2.9271677,  2.81272876, 0.57582833,
 2.21248718, 1.2276669,  0.17059022, 0.19674575, 1.6384496,  2.57059707,
 1.41351574, 2.74980657, 2.23813116, 0.2220626,  1.89595738, 1.81189179,
 0.86605858, 2.6563126,  0.61201975, 0.25312341, 2.707325,   0.67526109,
 2.29181584, 1.44308136, 2.26065017, 2.9183723,  0.60992381, 0.56408085,
 2.59574789, 0.05025349, 1.07762439, 0.01770534, 1.41920369, 2.32996914,
 2.89557076, 3.13212856, 2.85096232, 1.7391249 ]
def q_10():
    x = gen_set()
    print("Error with no noise: " + str(test_data(x)))
    print("Error with 1 sig noise: " + str(test_data(gen_noisy_set(1.0))))
    print("Error with 2 sig noise: " + str(test_data(gen_noisy_set(2.0))))
    print("Error with 8 sig noise: " + str(test_data(gen_noisy_set(8.0))))
    print("Error with 24 sig noise: " + str(test_data(gen_noisy_set(24.0))))
    print("Error with 64 sig noise: " + str(test_data(gen_noisy_set(64.0))))

def test_data(x):
    y = [np.sin(val) for val in x]
    poly = sc.lagrange(x, y)
    err = 0
    for test in test_x_points:
        err += np.mean((np.polyval(poly, test) - y) ** 2)
    return np.log(err)

def gen_set():
    return np.random.uniform(0,np.pi,size = 100)

def gen_noisy_set(sig):
    return np.random.normal(0,sig,size=100) + np.random.uniform(0,np.pi,size=100)
def main():
    # Use a breakpoint in the code line below to debug your script.
    q_10()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


