import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold

from scipy import misc
import os









# Look pretty...
# matplotlib.style.use('ggplot')
plt.style.use('ggplot')


#
# TODO: Start by creating a regular old, plain, "vanilla"
# python list. You can call it 'samples'.
#
samples = [] 


# TODO: Write a for-loop that iterates over the images in the
# Module4/Datasets/ALOI/32/ folder, appending each of them to
# your list. Each .PNG image should first be loaded into a
# temporary NDArray, just as shown in the Feature
# Representation reading.
#
# Optional: Resample the image down by a factor of two if you
# have a slower computer. You can also convert the image from
# 0-255  to  0.0-1.0  if you'd like, but that will have no
# effect on the algorithm's results.
#


#indir = 'Datasets/ALOI/32/'
#for root, dirs, filenames in os.walk(indir):
#    for f in filenames:
#        print(indir +f)
#        log = open(indir + f, 'r')

path = "Datasets/ALOI/32/"
for root, dirs, filenames in os.walk(path):
    for file in filenames:
        img = misc.imread(path+'/{0}'.format(file))
        img = ((img[::2, ::2]/250.0).reshape(-1))
        samples.append(img)


# TODO: Once you're done answering the first three questions,
# right before you converted your list to a dataframe, add in
# additional code which also appends to your list the images
# in the Module4/Datasets/ALOI/32_i directory. Re-run your
# assignment and answer the final question below.
#
#
# TODO: Convert the list to a dataframe
#
df=pd.DataFrame(samples)
#
# TODO: Implement Isomap here. Reduce the dataframe df down
# to three components, using K=6 for your neighborhood size
#
# .. your code here .. 

iso = manifold.Isomap(n_neighbors=2, n_components=3)
iso.fit(df)
manifold.Isomap(eigen_solver='auto', max_iter=None, n_components=3, n_neighbors=2,
    neighbors_algorithm='auto', path_method='auto', tol=0)
manifold = iso.transform(df)


#
# TODO: Create a 2D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker. Graph the first two
# isomap components
#
# .. your code here .. 
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('2D ISOMAP')
ax.set_xlabel('Component: {0}'.format(0))
ax.set_ylabel('Component: {0}'.format(1))
ax.scatter(manifold[:,0],manifold[:,1], marker='o',alpha=0.7)

#
# TODO: Create a 3D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker:
#
fig2 = plt.figure()
ax1 = fig2.add_subplot(111,projection='3d')
ax1.set_title('3D ISOMAP')
ax1.set_xlabel('Component: {0}'.format(0))
ax1.set_ylabel('Component: {0}'.format(1))
ax1.set_zlabel('Component: {0}'.format(2))


ax1.scatter(manifold[:,0],manifold[:,1],manifold[:,2], marker='o',alpha=0.7)
plt.show()

