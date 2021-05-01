import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

print('Implementation of K-means method to three Gaussians')
print('Enter the number of points for each Gaussian:')
N = int(input())

### Generate 3 groups of Gaussian numbers with different mean
x1 = randn(N)
y1 = randn(N)

x2 = 3.+randn(N)
y2 = 3.+randn(N)

x3 = 3.+randn(N)
y3 = randn(N)

x = np.append(np.append(x1,x2),x3)
y = np.append(np.append(y1,y2),y3)

fig, (ax1, ax2)  = plt.subplots(1, 2)

ax1.scatter(x,y,c='black',s=0.5)
ax1.set_title('Non-clustered')
data = np.transpose([x,y])

#### use of the K-means method implemented in sklearn
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)
y_kmeans = kmeans.predict(data)
ax2.scatter(data[:, 0], data[:, 1], c=y_kmeans, s=0.5, cmap='viridis')
centers = kmeans.cluster_centers_
ax2.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5)
ax2.set_title('K-means clustered')
plt.show()


