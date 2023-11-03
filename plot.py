import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('darkgrid')
x = [0, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1]

maxpos = [67.17, 65.53, 65.62, 54.52, 52.92, 54.11, 46.44, 37.4, 36.21]

maxneg = [67.17, 67.95, 64.11, 60.5, 57.58, 54.89, 47.76, 36.07, 36.21]

ourspos = [67.17, 53.61, 47.53, 44.2, 43.88, 43.79, 42.74, 33.15, 36.21]

oursneg = [67.17, 66.16, 63.38, 62.47, 60.55, 55.25, 51.6, 45.98, 36.21]

# plt.title('Positive perturbation test for 50 COCOval images')
# plt.plot(x, maxpos)
# plt.plot(x, ourspos)
# plt.legend(['DSM', 'RM']) 
# plt.show()

plt.title('Negative perturbation test for 50 COCOval images')
plt.plot(x, maxneg)
plt.plot(x, oursneg)
plt.legend(['DSM', 'RM']) 
plt.show()