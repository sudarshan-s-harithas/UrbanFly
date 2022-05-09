import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

means = [2,1,0.5,0]
covariances = [1, 0.6, 0.2]
errors = np.array([])

NUM_OF_GAUSSIANS = 4

# Fit GMM
gmm = GaussianMixture(n_components = NUM_OF_GAUSSIANS, covariance_type='spherical')

# Plot histograms and gaussian curves
# fig, ax = plt.subplots()
# ax.hist(img.ravel(),255,[2,256], normed=True)

fig, ax = plt.subplots()
for var_id, variance in enumerate(covariances[:2]):
    for mean_id, mean in enumerate(means[-2:]):
        # for i in range(50):
        #     errors = np.concatenate((errors, self.get_error_sdfs(mean=mean, variance=variance)))

        errors = np.loadtxt("histogram_data_non_gauss_"+str(var_id)+"_"+str(mean_id)+".txt")
        errors = errors[errors<5]

        hist, bins = np.histogram(errors, bins=50, density=True)
        width = (1/len(means)) * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        ax.bar(center + (mean_id/len(means)) * 0.5, hist, align='center', width=width, alpha=0.4, label="variance = " + str(variance) + ", mean (m) ="+str(mean))

        gmm = gmm.fit(X=np.expand_dims(errors,1))

gmm_x = np.linspace(0,np.max(errors),256)
gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1,1)))

for var_id, variance in enumerate(covariances[:2]):
    # Evaluate GMM
    ax.plot(gmm_x, gmm_y, lw=4, label="GMM_var_"+str(covariances[var_id]))
# ax[var_id].legend(loc='upper right')

print("GMM means are : ", gmm.means_)
print("GMM covariances are : ", gmm.covariances_)
print("GMM weights are : ", gmm.weights_)

np.savetxt('GMM_'+str(NUM_OF_GAUSSIANS)+'_means.txt', gmm.means_.reshape((1, NUM_OF_GAUSSIANS)), fmt='%g')
np.savetxt('GMM_'+str(NUM_OF_GAUSSIANS)+'_covariances.txt', gmm.covariances_.reshape((1, NUM_OF_GAUSSIANS)), fmt='%g')
np.savetxt('GMM_'+str(NUM_OF_GAUSSIANS)+'_weights.txt', gmm.weights_.reshape((1, NUM_OF_GAUSSIANS)), fmt='%g')

plt.title('Probability distribution of plane sdf error w.r.t ground truth sdf')
plt.show()