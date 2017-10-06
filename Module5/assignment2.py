import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans

matplotlib.style.use('ggplot') # Look Pretty



  # exit()# .. your code here ..




#
# INFO: This dataset has call records for 10 users tracked over the course of 3 years.
# Your job is to find out where the users likely live and work at!


#
# TODO: Load up the dataset and take a peek at its head
# Convert the date using pd.to_datetime, and the time using pd.to_timedelta
#

df = pd.read_csv('Datasets/CDR.csv')
df[['CallDate']] = df[['CallDate']].apply(pd.to_datetime,errors='coerce')
df[['CallTime']] = df[['CallTime']].apply(pd.to_timedelta,errors='coerce')

#
# TODO: Get a distinct list of "In" phone numbers (users) and store the values in a
# regular python list.
# Hint: https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.tolist.html
#
# .. your code here ..

users = df.In.unique().tolist()

# 
# TODO: Create a slice called user1 that filters to only include dataset records where the
# "In" feature (user phone number) is equal to the first number on your unique list above;
# that is, the very first number in the dataset
#
for  phone_number in users:
    user = df[(df['In']==phone_number) & ((df['DOW']=='Sat') | (df['DOW']=='Sun'))]
    print phone_number
    # INFO: Plot all the call locations
    user.plot.scatter(x='TowerLon', y='TowerLat', c='gray', alpha=0.9, title='Call Locations')
    #showandtell()  # Comment this line out when you're ready to proceed
    
    #
    # INFO: The locations map above should be too "busy" to really wrap your head around. This
    # is where domain expertise comes into play. Your intuition tells you that people are likely
    # to behave differently on weekends:
    #
    # On Weekends:
    #   1. People probably don't go into work
    #   2. They probably sleep in late on Saturday
    #   3. They probably run a bunch of random errands, since they couldn't during the week
    #   4. They should be home, at least during the very late hours, e.g. 1-4 AM
    #
    # On Weekdays:
    #   1. People probably are at work during normal working hours
    #   2. They probably are at home in the early morning and during the late night
    #   3. They probably spend time commuting between work and home everyday
    
    
    
    #
    # TODO: Add more filters to the user1 slice you created. Add bitwise logic so that you're
    # only examining records that came in on weekends (sat/sun).
    #
    
    #
    # TODO: Further filter it down for calls that came in either before 6AM OR after 10pm (22:00:00).
    # You can use < and > to compare the string times, just make sure you code them as military time
    # strings, eg: "06:00:00", "22:00:00": https://en.wikipedia.org/wiki/24-hour_clock
    #
    # You might also want to review the Data Manipulation section for this. Once you have your filtered
    # slice, print out its length:
    #
    user = user[(user['CallTime']>'06:00:00') & (user['CallTime']<'22:00:00')]
    
       
    #
    # INFO: Visualize the dataframe with a scatter plot as a sanity check. Since you're familiar
    # with maps, you know well that your X-Coordinate should be Longitude, and your Y coordinate
    # should be the tower Latitude. Check the dataset headers for proper column feature names.
    # https://en.wikipedia.org/wiki/Geographic_coordinate_system#Geographic_latitude_and_longitude
    #
    # At this point, you don't yet know exactly where the user is located just based off the cell
    # phone tower position data; but considering the below are for Calls that arrived in the twilight
    # hours of weekends, it's likely that wherever they are bunched up is probably near where the
    # caller's residence:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(user.TowerLon,user.TowerLat, c='g', marker='o', alpha=.9)
    ax.set_title('Weekend Calls (<6am or >10p)')
    #showandtell()  # TODO: Comment this line out when you're ready to proceed
    
    
    
    #
    # TODO: Run K-Means with a K=1. There really should only be a single area of concentration. If you
    # notice multiple areas that are "hot" (multiple areas the usr spends a lot of time at that are FAR
    # apart from one another), then increase K=2, with the goal being that one of the centroids will
    # sweep up the annoying outliers; and the other will zero in on the user's approximate home location.
    # Or rather the location of the cell tower closest to their home.....
    #
    # Be sure to only feed in Lat and Lon coordinates to the KMeans algo, since none of the other
    # data is suitable for your purposes. Since both Lat and Lon are (approximately) on the same scale,
    # no feature scaling is required. Print out the centroid locations and add them onto your scatter
    # plot. Use a distinguishable marker and color.
    #
    # Hint: Make sure you graph the CORRECT coordinates. This is part of your domain expertise.
    #
    newdf=user[['TowerLat','TowerLon']]
    model = KMeans(n_clusters=1)
    model.fit(newdf)
    KMeans(copy_x=True, init='k-means++', max_iter=300, n_clusters=1, n_init=10,
        n_jobs=1, precompute_distances='auto', random_state=None, tol=0.0001,
        verbose=0)
      # INFO: Print and plot the centroids...
    labels = model.predict(newdf)
    centroids = model.cluster_centers_
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Kmeans - Weekend Calls (<6am or >10p)')
    ax1.scatter(centroids[:,0], centroids[:,1], marker='x', c='red', alpha=0.5, linewidths=1, s=169)
    print centroids
    #exit()
    plt.show()
    #showandtell()  # TODO: Comment this line out when you're ready to proceed



#
# TODO: Repeat the above steps for all 10 individuals, being sure to record their approximate home
# locations. You might want to use a for-loop, unless you enjoy typing.
#
# .. your code here ..

