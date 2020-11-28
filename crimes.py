"""
Program to determine the worst placees to park in the City of Orlando.

This is determined based off crime reports for Vehicle theft.

Author:  Brenden Morton
Date started: 11/22/2020
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import seaborn as sns
from collections import OrderedDict
#import matplotlib.pyplot as plt
from itertools import cycle
import re
import os

import markov_clustering as mcl
import networkx as nx
import random


os.chdir(r'C:\Users\Brenden Morton\Desktop\UCF\Fall2020\Network Optimization\Final Project')

# Read in CSV
crime_data = pd.read_csv("OPD_Crimes.csv")

# Replace spaces with underscores (_)
crime_data.columns = [column.replace(" ","_") for column in crime_data.columns]

# Only show Vehicle Theft cases
crime_filterd = crime_data.query('Case_Offense_Category == "Vehicle Theft"')

# Crime data of all crimes (No Filtering)
all_crimes = crime_data
all_crimes = all_crimes[['Case_Location','Location','Case_Offense_Category', 'Case_Offense_Type']]
all_crimes.reset_index(drop=True, inplace=True)

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

# Filter out crimes without geographical data (Long., Lat.)
all_crimes = all_crimes.dropna()
crime_filterd = crime_filterd.dropna()

# Strip paranethesesisisis off string
def slice_me_up_baby(y_or_something):
    s = slice(1, len(y_or_something)-1)
    y_or_something = y_or_something[s]
    # print(y_or_something)
    return y_or_something

crime_filterd.Location = crime_filterd.Location.apply(lambda x: slice_me_up_baby(x))

crime_filterd['Longitude'] = crime_filterd.Location.apply(lambda x: x.split(', ')[0])
crime_filterd['Latitude'] = crime_filterd.Location.apply(lambda x: x.split(', ')[1])


print('\n\t\t\t\t\tLongitude: 28.xxxx \t Latitude -81.xxxx\n')
#crime_filterd['Longitude'] = crime_filterd.Longitude.apply(lambda x: x.split('.')[1])
#crime_filterd['Latitude'] = crime_filterd.Latitude.apply(lambda x: x.split('.')[1])

crime_filterd['Longitude'] = crime_filterd.Longitude.apply(lambda x: float(x))
crime_filterd['Latitude'] = crime_filterd.Latitude.apply(lambda x: float(x))


crime_filterd = crime_filterd[["Longitude", "Latitude"]]

bandwidth = estimate_bandwidth(crime_filterd, quantile=0.15, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(crime_filterd)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)


bigP = np.array(crime_filterd)

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

writer = pd.ExcelWriter(r'C:\Users\Brenden Morton\Desktop\UCF\Fall2020\Network Optimization\Final Project\clusters_mean_shift.xlsx')
crime_filterd.to_excel(writer)
writer.save()

#------------------------------------------------------------------------------------------

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



# Split address on /
streets_filtered["Street_1"] = streets_filtered.Case_Location.apply(lambda x: x.split('/')[0])
streets_filtered["Street_2"] = streets_filtered.Case_Location.apply(lambda x: x.split('/')[1])


# Trim leading and lagging spaces
streets_filtered["Street_1"] = streets_filtered.Street_1.apply(lambda x: x[:-1])
streets_filtered["Street_2"] = streets_filtered.Street_2.apply(lambda x: x[1:])


# Make series
streets_1 = streets_filtered.Street_1
streets_2 = streets_filtered.Street_2

# Looks good to me - BAiley
streets_1 = streets_1.str.replace("AV", "AVE", regex=False)
streets_2 = streets_2.str.replace("AV", "AVE", regex=False)
streets_1 = streets_1.str.replace("AVEE", "AVE", regex=False)
streets_2 = streets_2.str.replace("AVEE", "AVE", regex=False)

# Looks good to me - BAiley
streets_1 = streets_1.str.replace("BOGGY CREE", "BOGGY CREEK RD", regex=False)
streets_2 = streets_2.str.replace("BOGGY CREE", "BOGGY CREEK RD", regex=False)
streets_1 = streets_1.str.replace("BOGGY CREEK RDK RD", "BOGGY CREEK RD", regex=False)
streets_2 = streets_2.str.replace("BOGGY CREEK RDK RD", "BOGGY CREEK RD", regex=False)


# Concat both sets of streets
#streets = pd.concat([streets_filtered.Street_1, streets_filtered.Street_2], ignore_index=True)
streets = pd.concat([streets_1, streets_2], ignore_index=True)

# Remove duplicates
streets_unique = streets.unique()
#streets_unique = streets

# Initializing adjacency matrix of (unique streets x unique streets) with zeros for all entries
matrix = np.zeros((len(streets_unique),len(streets_unique)))

w = []
for street in streets_unique:
    if "BOGGY CREE" in street:
        street = 'BOGGY CREEK RD'
        w.append(street)
    else:
        w.append(street)
streets_unique = w

street_map_1 = {}
street_map_2 = {}

for str1,str2,i in zip(streets_1, streets_2, range(len(streets_unique))):
    street_map_1.update({str1:i})
    street_map_2.update({str2:i})
    
streets_1 = pd.DataFrame(streets_1)
streets_1['Mapping'] = range(0,401)

streets_2 = pd.DataFrame(streets_2)
streets_2['Mapping'] = range(0,401)
#street_mapping = OrderedDict(sorted(street_mapping.items(), key=lambda x: x[1]))
#print(street_mapping)
#print(street_mapping[re.findall("S TAMPA AVE")]
#street_mapping = dict(street_mapping)

# Filling in adjacency matrix
#for i,j in zip(street_map_1.keys(), street_map_2.keys()):
#   matrix[street_map_1[i]][street_map_2[j]] +=1
 
for i,j in zip(street_map_1.keys(), street_map_2.keys()):
    matrix[street_map_1[i]][street_map_2[j]] +=1   
 




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


#-------------Markov Clustering for Street Location data of Motor Vehicle Theft Crimes-------------------------


# making tuples for our data for  me and my brither brother is what i meant but i said brither =

my_tuple = {}

for i,j,k  in zip(street_map_1.keys(), street_map_2.keys(), range(len(street_map_1))):
    my_tuple.update({k:(street_map_1[i],street_map_2[j])})

#street_map_1[streets_1[i]]

# number of nodes to use
numnodes = 192

# generate random positions as a dictionary where the key is the node id and the value
# is a tuple containing 2D coordinates
#positions = {i:(random.random() * 2 - 1, random.random() * 2 - 1) for i in range(numnodes)}

# use networkx to generate the graph
#network = nx.random_geometric_graph(numnodes, 0.3, pos=positions)

# then get the adjacency matrix (in sparse form)
#matrix = nx.to_scipy_sparse_matrix(network)


zatrix = nx.from_numpy_matrix(matrix)

print("Node of graph: ")
print(zatrix.nodes())
print("Edges of graph: ")
print(zatrix.edges())
print("info: ")
print(nx.info(zatrix, n=None))

ratrix = zatrix
ratrix = nx.to_numpy_matrix(ratrix)
res = mcl.run_mcl(ratrix)
clusters = mcl.get_clusters(res)

mcl.drawing.draw_graph(ratrix, clusters, edge_color="red",node_size=30, with_labels=False)
#---------------------------------------------------------------------------------------------------------------

"""
nx.draw_networkx(zatrix)


positions = my_tuple
natrix = nx.to_scipy_sparse_matrix(zatrix)




result = mc.run_mcl(natrix)           # run MCL with default parameters
clusters = mc.get_clusters(result)    # get clusters

mc.draw_graph(natrix, clusters, pos=positions, node_size=5, with_labels=False, edge_color="silver")
"""
"""
result = mc.run_mcl(matrix, inflation=1.4)
clusters = mc.get_clusters(result)
mc.draw_graph(matrix, clusters, pos=positions, node_size=50, with_labels=False, edge_color="silver")
"""
#-----------------------------------

import time
start_time = time.time()


all_crimes.Location = all_crimes.Location.apply(lambda x: slice_me_up_baby(x))

all_crimes['Longitude'] = all_crimes.Location.apply(lambda x: x.split(', ')[0])
all_crimes['Latitude'] = all_crimes.Location.apply(lambda x: x.split(', ')[1])


print('\n\t\t\t\t\tLongitude: 28.xxxx \t Latitude -81.xxxx\n')
#crime_filterd['Longitude'] = crime_filterd.Longitude.apply(lambda x: x.split('.')[1])
#crime_filterd['Latitude'] = crime_filterd.Latitude.apply(lambda x: x.split('.')[1])

all_crimes['Longitude'] = all_crimes.Longitude.apply(lambda x: float(x))
all_crimes['Latitude'] = all_crimes.Latitude.apply(lambda x: float(x))

all_crimes['Longitude'].values[0]

#all_crimes = all_crimes[["Longitude", "Latitude"]]

def filter_nw(x, col):
    for street in x[col]:
        if "/" not in street:
            x = x[x[col] != street]
    return x

# all_streets = crime_data.query('Case_Offense_Category == "Vehicle Theft"')
#all_streets_for_all_crimes = crime_data.query('Case_Offense_Category == "Vehicle Theft"')

all_streets_for_all_crimes = crime_data.query('Case_Offense_Category == "Vehicle Theft" | Case_Offense_Category == "Theft" | Case_Offense_Category == "Burglary"')

all_streets_for_all_crimes.reset_index(drop=True, inplace=True)
filtered = filter_nw(all_streets_for_all_crimes, 'Case_Location')



# Split address on /
filtered["Street_1"] = filtered.Case_Location.apply(lambda x: x.split('/')[0])
filtered["Street_2"] = filtered.Case_Location.apply(lambda x: x.split('/')[1])


# Trim leading and lagging spaces
filtered["Street_1"] = filtered.Street_1.apply(lambda x: x[:-1])
filtered["Street_2"] = filtered.Street_2.apply(lambda x: x[1:])

filtered = filtered.dropna()

filtered.Location = filtered.Location.apply(lambda x: slice_me_up_baby(x))

filtered['Longitude'] = filtered.Location.apply(lambda x: x.split(', ')[0])
filtered['Latitude'] = filtered.Location.apply(lambda x: x.split(', ')[1])

filtered['Longitude'] = filtered.Longitude.apply(lambda x: float(x))
filtered['Latitude'] = filtered.Latitude.apply(lambda x: float(x))


# Make series
streets_1 = filtered.Street_1
streets_2 = filtered.Street_2


streets_1 = streets_1.str.replace("AV", "AVE", regex=False)
streets_2 = streets_2.str.replace("AV", "AVE", regex=False)
streets_1 = streets_1.str.replace("AVEE", "AVE", regex=False)
streets_2 = streets_2.str.replace("AVEE", "AVE", regex=False)

streets_1 = streets_1.str.replace("BLVD", "BV", regex=False)
streets_2 = streets_2.str.replace("BLVD", "BV", regex=False)

streets_1 = streets_1.str.replace("EBO", "E", regex=False)
streets_2 = streets_2.str.replace("EBO", "E", regex=False)
streets_1 = streets_1.str.replace("EB", "E", regex=False)
streets_2 = streets_2.str.replace("EB", "E", regex=False)

streets_1 = streets_1.str.replace("WBO", "W", regex=False)
streets_2 = streets_2.str.replace("WBO", "w", regex=False)
streets_1 = streets_1.str.replace("WB", "W", regex=False)
streets_2 = streets_2.str.replace("WB", "W", regex=False)

streets_1 = streets_1.str.replace("I4", "I-4", regex=False)
streets_2 = streets_2.str.replace("I4", "I-4", regex=False)

streets_1 = streets_1.str.replace("TL", "TRL", regex=False)
streets_2 = streets_2.str.replace("TL", "TRL", regex=False)

streets_1 = streets_1.str.replace("PKWY", "PY", regex=False)
streets_2 = streets_2.str.replace("PKWY", "PY", regex=False)

#streets_1 = streets_1.str.replace("AVEE", "AVE", regex=False)
#streets_2 = streets_2.str.replace("AVEE", "AVE", regex=False)




streets = pd.concat([streets_1, streets_2], ignore_index=True)

# Remove duplicates
streets_unique = streets.unique()

#streets_unique_df = pd.DataFrame(streets_unique) 
#writer = pd.ExcelWriter(r'C:\Users\Brenden Morton\Desktop\UCF\Fall2020\Network Optimization\Final Project\modifed_streets_cleaned.xlsx')
#streets_unique_df.to_excel(writer)
#writer.save()


street_map_1 = {}
street_map_2 = {}

street_mapping = {}
#for str1,str2,i in zip(streets_1, streets_2, range(len(streets_unique))):
#   street_map_1.update({str1:i})
#   street_map_2.update({str2:i})
 
for street,i in zip(streets_unique, range(len(streets_unique))):
    street_mapping.update({street:i})
   
#streets_1 = pd.DataFrame(streets_1)
#streets_1['Mapping'] = range(0,401)

#streets_2 = pd.DataFrame(streets_2)
#streets_2['Mapping'] = range(0,401)
 

all_theft_matrix = np.zeros((len(streets_unique),len(streets_unique)))


#for i,j in zip(street_map_1.keys(), street_map_2.keys()):
#    all_theft_matrix[street_map_1[streets_1[i]]][street_map_2[streets_2[j]]] += 1 

for i,j in zip(streets_1, streets_2):
    all_theft_matrix[street_mapping[i]][street_mapping[j]] += 1
    all_theft_matrix[street_mapping[j]][street_mapping[i]] += 1


print("--- %s seconds ---" % (time.time() - start_time))


# filtered_tuple = {}

positions = {}

print(all_theft_matrix.sum())

zatrix = nx.from_numpy_matrix(all_theft_matrix)

print("Node of graph: ")
print(zatrix.nodes())
print("Edges of graph: ")
print(zatrix.edges())
print("info: ")
print(nx.info(zatrix, n=None))

ratrix = zatrix
ratrix = nx.to_numpy_matrix(ratrix)
res = mcl.run_mcl(ratrix)
clusters = mcl.get_clusters(res)

mcl.drawing.draw_graph(ratrix, clusters, edge_color="red",node_size=0.5, with_labels=False)

# export graph to cytoscape format

nx.write_gml(zatrix, "OrlandoCrimes_Theft.gml")
























