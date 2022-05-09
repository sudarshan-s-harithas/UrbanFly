from distutils.log import error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

NUM_OF_GAUSSIANS = 4

# Fit GMM
gmm = GaussianMixture(n_components = NUM_OF_GAUSSIANS, covariance_type='spherical')

data = np.loadtxt("plane_error.txt")

# data = data[data[:, 4] > 10]
# data = data[data[:, 3] < 20]
# data = data[data[:, 2] < 5]
errors = np.degrees(np.arccos(1-(1-data[:, 1:2])))

# plt.plot(errors)
# plt.xlabel("number of view points")
# plt.ylabel("absolute angle error (degrees) of normals")
# plt.show()

y, x = np.histogram(errors, bins=50, density=True)
plt.bar(x[:-1].flatten(), y.flatten())
plt.title("distribution of normal error")
plt.xlabel("absolute angle error (degrees) of normals")
plt.ylabel("number of plane estimates")
plt.show()

gmm = gmm.fit(X=np.expand_dims(np.concatenate((errors.flatten(), -errors.flatten())),1))

gmm_x = np.linspace(-np.max(errors),np.max(errors),256)
gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1,1)))
plt.plot(gmm_x, gmm_y, lw=4, label="normal gmm distro")
plt.show()

np.savetxt('normal_GMM_'+str(NUM_OF_GAUSSIANS)+'_means.txt', gmm.means_.reshape((1, NUM_OF_GAUSSIANS)), fmt='%g')
np.savetxt('normal_GMM_'+str(NUM_OF_GAUSSIANS)+'_covariances.txt', gmm.covariances_.reshape((1, NUM_OF_GAUSSIANS)), fmt='%g')
np.savetxt('normal_GMM_'+str(NUM_OF_GAUSSIANS)+'_weights.txt', gmm.weights_.reshape((1, NUM_OF_GAUSSIANS)), fmt='%g')

# plt.bar(data[:, 4].flatten(), errors.flatten())
# plt.show()

errors = data[:, 2:3]

# plt.plot(errors)
# plt.xlabel("number of view points")
# plt.ylabel("plane offset error (meters)")
# plt.show()

y, x = np.histogram(errors)
plt.plot(x[:-1], y)
plt.title("distribution of plane offset error")
plt.xlabel("plane offset error")
plt.ylabel("number of plane estimates")
plt.show()

gmm = gmm.fit(X=np.expand_dims(np.concatenate((errors.flatten(), -errors.flatten())),1))

gmm_x = np.linspace(-np.max(errors),np.max(errors),256)
gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1,1)))
plt.plot(gmm_x, gmm_y, lw=4, label="normal gmm distro")
plt.show()

np.savetxt('offset_GMM_'+str(NUM_OF_GAUSSIANS)+'_means.txt', gmm.means_.reshape((1, NUM_OF_GAUSSIANS)), fmt='%g')
np.savetxt('offset_GMM_'+str(NUM_OF_GAUSSIANS)+'_covariances.txt', gmm.covariances_.reshape((1, NUM_OF_GAUSSIANS)), fmt='%g')
np.savetxt('offset_GMM_'+str(NUM_OF_GAUSSIANS)+'_weights.txt', gmm.weights_.reshape((1, NUM_OF_GAUSSIANS)), fmt='%g')

plt.bar(data[:, 4].flatten(), errors.flatten())
plt.title("plane offset error vs number of triangulated points")
plt.xlabel("number of triangulated points")
plt.ylabel("plane offset error")
plt.show()

# plt.bar(data[:, 3].flatten(), errors.flatten())
# plt.title("plane offset error vs distance from true plane")
# plt.xlabel("distance from true plane")
# plt.ylabel("plane offset error")
# plt.show()

# plane_dict = {}
# for idx in set(data[:, 0]):
#     plane_dict[idx] = data[data[:, 0] == idx][:, 1:]

# print("Number of planes are ", len(plane_dict))
# print("total number of samples are ", data.shape[0])

# normals_errors = []
# offset_errors = []

# for idx in plane_dict.keys():
#     # print("number of samples with id ", idx, " are ", plane_dict[idx].shape[0])

#     min_normal_error = np.min(np.degrees(np.arccos(plane_dict[idx][:, 0])))
#     min_offset_error = np.min(plane_dict[idx][:, 1])

#     normals_errors.append(min_normal_error)
#     offset_errors.append(min_offset_error)


# errors = normals_errors

# # plt.plot(errors)
# # plt.xlabel("number of view points")
# # plt.ylabel("absolute angle error (degrees) of normals")
# # plt.show()

# y, x = np.histogram(errors, bins=np.arange(20), density=True)
# plt.plot(x[:-1], y)
# plt.title("distribution of normal error")
# plt.xlabel("absolute angle error (degrees) of normals")
# plt.ylabel("probability")
# plt.show()

# errors = offset_errors

# y, x = np.histogram(errors, bins=np.arange(10, step=0.1), density=True)
# plt.plot(x[:-1], y)
# plt.title("distribution of plane offset error")
# plt.xlabel("plane offset error")
# plt.ylabel("probability")
# plt.show()
