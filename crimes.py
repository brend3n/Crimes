"""
Program to determine the worst placees to park in the City of Orlando.

This is determined based off crime reports for Vehicle theft.

Author:  Brenden Morton
Date started: 11/22/2020
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import seaborn as sns

#import matplotlib.pyplot as plt
from itertools import cycle

# Read in CSV
crime_data = pd.read_csv("OPD_Crimes.csv")
# print(crime_data)

# Replace spaces with underscores (_)
crime_data.columns = [column.replace(" ","_") for column in crime_data.columns]

# Only show Vehicle Theft cases
crime_filterd = crime_data.query('Case_Offense_Category == "Vehicle Theft"')
# print(crime_filterd)

# Subset with only Case_Location and Location
crime_filterd = crime_filterd[['Case_Location', 'Location']]
crime_filterd.reset_index(drop=True, inplace=True)


# Convert Lat, Lon to UTM



# crime_filterd[str(x_coord)] = crime_filterd
# print(crime_filterd.Location.values[0].split(',')[0])

# crime_filterd.Location = crime_filterd.Location.apply(lambda x: tuple(x))
# print(crime_filterd.Location.values[0])
# print(crime_filterd.Location)

# print(type(crime_filterd.Location.values[0]))


crime_filterd = crime_filterd.dropna()


# Strip paranethesesisisis off string
def slice_me_up_baby(y_or_something):
    s = slice(1, len(y_or_something)-1)
    y_or_something = y_or_something[s]
    # print(y_or_something)
    return y_or_something


# slice_me_up_baby("(1231028398123.2342234)")
crime_filterd.Location = crime_filterd.Location.apply(lambda x: slice_me_up_baby(x))
# print(crime_filterd.Location)

crime_filterd['Longitude'] = crime_filterd.Location.apply(lambda x: x.split(', ')[0])
crime_filterd['Latitude'] = crime_filterd.Location.apply(lambda x: x.split(', ')[1])

# print(crime_filterd)



print('\n\t\t\t\t\tLongitude: 28.xxxx \t Latitude -81.xxxx\n')
#crime_filterd['Longitude'] = crime_filterd.Longitude.apply(lambda x: x.split('.')[1])
#crime_filterd['Latitude'] = crime_filterd.Latitude.apply(lambda x: x.split('.')[1])
print(crime_filterd)

crime_filterd['Longitude'] = crime_filterd.Longitude.apply(lambda x: float(x))
crime_filterd['Latitude'] = crime_filterd.Latitude.apply(lambda x: float(x))




def get_network_dataset(x):
    streets = [strt for strt in x if "/" in strt]
    return streets




def filter_nw(x, col):
    for street in x[col]:
        if "/" not in street:
            x = x[x[col] != street]
    return x

all_streets = crime_data.query('Case_Offense_Category == "Vehicle Theft"')
all_streets.reset_index(drop=True, inplace=True)
streets_filtered = filter_nw(all_streets, 'Case_Location')


streets_filtered["Street_1"] = streets_filtered.Case_Location.apply(lambda x: x.split('/')[0])
streets_filtered["Street_2"] = streets_filtered.Case_Location.apply(lambda x: x.split('/')[1])

streets = pd.concat([streets_filtered.Street_1, streets_filtered.Street_2], ignore_index=True)
streets_unique = streets.unique()







crime_filterd = crime_filterd[["Longitude", "Latitude"]]

bandwidth = estimate_bandwidth(crime_filterd, quantile=0.15, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(crime_filterd)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)


bigP = np.array(crime_filterd)
#
from itertools import cycle

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(bigP[my_members, 0], bigP[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Fungus my brungus: %d' % n_clusters_)
# plt.show()


crime_filterd["Labels"] = labels

"""
plt.figure(figsize=(15,5))
plt.scatter(crime_filterd.Longitude, crime_filterd.Latitude,marker=labels)
plt.xlabel = "Longitude"
plt.ylabel = "Latitude"
plt.title = "Crime Clusters Clings Clanks and Cranks"
#plt.show()
"""
sns.scatterplot('Longitude', 'Latitude', data=crime_filterd, hue='Labels')


"""
matrix = crime_filterd[["Longitude", "Latitude"]]
matrix = np.array(matrix)
#matrix = matrix.reshape(len(matrix), len(matrix))
matrix = matrix.reshape(1,-1)
"""
print("Its lit")



writer = pd.ExcelWriter(r'C:\Users\Brenden Morton\Desktop\UCF\Fall2020\Network Optimization\Final Project\clusters_mean_shift.xlsx')
crime_filterd.to_excel(writer)
writer.save()


"""
new flip flops $26
new Tradi Zori $80
------------------
$106  
    $63.6   --%40 off price

running shoes $75
--------
$138.6 -total



im buying myself
    VR headset (quest) $250
        GIVE ME $100
------------
$ 238.6 if given $100
$ 338.6 if given $200
$ 388.6 if given $250
"""

"""
bigP = np.zeros(shape=(len(crime_filterd),len(crime_filterd)))

#bigP = np.array(crime_filterd, ndmin = 11291)
adjacency_matrix = np.square(bigP)

from sklearn.cluster import SpectralClustering
sc = SpectralClustering(7, affinity='precomputed', n_init=100,
                        assign_labels='discretize')
labels = sc.fit_predict(bigP)
crime_filterd["labels"] = labels
sns.scatterplot('Longitude', 'Latitude', data=crime_filterd, hue='labels')

print("kite is kinda cool but i dont use it")
"""





