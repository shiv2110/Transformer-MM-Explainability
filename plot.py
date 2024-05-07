import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('darkgrid')
x = [0, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1]

# maxpos = [67.17, 65.53, 65.62, 54.52, 52.92, 54.11, 46.44, 37.4, 36.21]

# maxneg = [67.17, 67.95, 64.11, 60.5, 57.58, 54.89, 47.76, 36.07, 36.21]

# ourspos = [67.17, 53.61, 47.53, 44.2, 43.88, 43.79, 42.74, 33.15, 36.21]

# oursneg = [67.17, 66.16, 63.38, 62.47, 60.55, 55.25, 51.6, 45.98, 36.21]

sai = [65.88, 54.52, 52.36, 51.28, 46.96, 44.72, 42.08, 36.68, 41.72]


# Absolute fiedler eigenvectors (lang ve+ no SEP)
sat = [65.88, 43.84, 29.88, 14.72, 11.48, 7.76, 5.92, 5.92, 5.92]

# Hila Chefer RMs (image ve+)
hci = [65.88, 51.76, 47.96, 44.72, 41.04, 41.84, 39.08, 35.6, 41.72]

# Hila Chefer RMs (lang ve+)
hct = [65.88, 31.56, 12.12, 4.96, 3.64, 3.0, 5.92, 5.92, 5.92]

# Transformer attribution (image ve+)
tai = [65.88, 55.88, 54.32, 50.96, 48.16, 46.04, 43.76, 38.68, 41.72]

# Transformer attribution (lang ve+)
tat = [65.88, 39.32, 28.76, 22.56, 16.72, 12.12, 5.92, 5.92, 5.92]


# plt.title('Positive perturbation test on image modality')
# plt.plot(x, sai)
# plt.plot(x, hci)
# plt.plot(x, tai)

plt.title('Positive perturbation test on text modality')
plt.plot(x, sai)
plt.plot(x, hci)
plt.plot(x, tai)

plt.legend(['DSM', 'RM', 'TA']) 
plt.xlabel('Fraction of tokens removed')
plt.ylabel('Accuracy')
plt.show()