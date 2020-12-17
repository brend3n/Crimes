"""
Program to determine the the most dangerous streets in the City of Orlando through the use of 
network clustering, data clustering techniques as well as network centrality analysis of the network.

This is determined based off crime reports for Vehicle theft.

Author:  Brenden Morton
Date started: 11/22/2020
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import seaborn as sns
from itertools import cycle
import os
import time
import csv
import markov_clustering as mcl
import networkx as nx


# Move to the proper directory where the source data is (OPD_crimes.csv)
os.chdir(r'C:\Users\Brenden Morton\Desktop\UCF\Fall2020\Network Optimization\Final Project\Source Data')

# Read in CSV of the source data
crime_data = pd.read_csv("OPD_Crimes.csv")

# Replace spaces with underscores (_)
crime_data.columns = [column.replace(" ","_") for column in crime_data.columns]

# Only show Vehicle Theft & Theft & Burglary cases
crime_filterd = crime_data.query('Case_Offense_Category == "Vehicle Theft" | Case_Offense_Category == "Theft" | Case_Offense_Category == "Burglary"')
#crime_filterd = crime_data.query('Case_Offense_Category == "Vehicle Theft"')


# Crime data of all crimes (No Filtering)
# Used for seeing all of the crime data
all_crimes = crime_data
all_crimes = all_crimes[['Case_Location','Location','Case_Offense_Category', 'Case_Offense_Type']]

# Reset the indices of the data
all_crimes.reset_index(drop=True, inplace=True)

# Subset the filtered data with only Case_Location and Location
crime_filterd = crime_filterd[['Case_Location', 'Location']]

# Reset the indices of the data
crime_filterd.reset_index(drop=True, inplace=True)

# Filter out crimes without geographical data (Long., Lat.)
all_crimes = all_crimes.dropna()
crime_filterd = crime_filterd.dropna()

# Strip paranethesesisisis off string
def slice_parentheses(y_or_something):
    s = slice(1, len(y_or_something)-1)
    y_or_something = y_or_something[s]
    return y_or_something

# Remove parentheses at the beginning and at the end of the string ("X","Y") -> "X", "Y"
crime_filterd.Location = crime_filterd.Location.apply(lambda x: slice_parentheses(x))

# Separate the Longitude and Latitude data and assign them to columns in the dataframe
crime_filterd['Longitude'] = crime_filterd.Location.apply(lambda x: x.split(', ')[0])
crime_filterd['Latitude'] = crime_filterd.Location.apply(lambda x: x.split(', ')[1])

# Cast the strings into their corresponding float data
crime_filterd['Longitude'] = crime_filterd.Longitude.apply(lambda x: float(x))
crime_filterd['Latitude'] = crime_filterd.Latitude.apply(lambda x: float(x))

# Subset the filtered data with Longitude and Latitude
crime_filterd = crime_filterd[["Longitude", "Latitude"]]

# Data Processing for intializing data for Mean Shift algorithm
bandwidth = estimate_bandwidth(crime_filterd, quantile=0.15, n_samples=500)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(crime_filterd)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
cf_arr = np.array(crime_filterd)

# Plotting the data
plt.figure(1)
plt.clf()

# Colors for the clusters
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')

# Coloring the clusters
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(cf_arr[my_members, 0], cf_arr[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)

# Title for the data
plt.title('Number of Clusters: %d' % n_clusters_)

# Add labels for the filtered data
crime_filterd["Labels"] = labels

# Plot the data
sns.scatterplot('Longitude', 'Latitude', data=crime_filterd, hue='Labels')

# Write the cluster data to a file for saving as well as exporting data to excel spreadsheet for storage
writer = pd.ExcelWriter(r'C:\Users\Brenden Morton\Desktop\UCF\Fall2020\Network Optimization\Final Project\clusters_mean_shift.xlsx')
crime_filterd.to_excel(writer)
writer.save()

# ^ all for Mean Shift clustering
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# Starting time used for timing the duration of the script
start_time = time.time()

# Remove parentheses at the beginning and at the end of the string ("X","Y") -> "X", "Y"
all_crimes.Location = all_crimes.Location.apply(lambda x: slice_parentheses(x))

# Separate the Longitude and Latitude data and assign them to columns in the dataframe
all_crimes['Longitude'] = all_crimes.Location.apply(lambda x: x.split(', ')[0])
all_crimes['Latitude'] = all_crimes.Location.apply(lambda x: x.split(', ')[1])

# Cast the strings into their corresponding float data
all_crimes['Longitude'] = all_crimes.Longitude.apply(lambda x: float(x))
all_crimes['Latitude'] = all_crimes.Latitude.apply(lambda x: float(x))


# Filter out any generalized theft crimes that are of the form [Block of X street]
# The remaining data should be of the form [street X / street Y]
def filter_nw(x, col):
    for street in x[col]:
        if "/" not in street:
            x = x[x[col] != street]
    return x

# Querying the data to get all of the unique
all_streets_for_all_crimes = crime_data.query('Case_Offense_Category == "Vehicle Theft" | Case_Offense_Category == "Theft" | Case_Offense_Category == "Burglary"')

# Resetting the index such that the first element starts at 0
all_streets_for_all_crimes.reset_index(drop=True, inplace=True)

# Filtering out invalid crime data formatting from dataset
filtered = filter_nw(all_streets_for_all_crimes, 'Case_Location')


# Splitting the string to tokenize the Longitude and Latitude
filtered["Street_1"] = filtered.Case_Location.apply(lambda x: x.split('/')[0])
filtered["Street_2"] = filtered.Case_Location.apply(lambda x: x.split('/')[1])

# Trim leading and lagging spaces
filtered["Street_1"] = filtered.Street_1.apply(lambda x: x[:-1])
filtered["Street_2"] = filtered.Street_2.apply(lambda x: x[1:])

# Drop null data
filtered = filtered.dropna()

# Remove parentheses at the beginning and at the end of the string ("X","Y") -> "X", "Y"
filtered.Location = filtered.Location.apply(lambda x: slice_parentheses(x))

# Separate the Longitude and Latitude data and assign them to columns in the dataframe
filtered['Longitude'] = filtered.Location.apply(lambda x: x.split(', ')[0])
filtered['Latitude'] = filtered.Location.apply(lambda x: x.split(', ')[1])

# Cast the strings into their corresponding float data
filtered['Longitude'] = filtered.Longitude.apply(lambda x: float(x))
filtered['Latitude'] = filtered.Latitude.apply(lambda x: float(x))


# Make series
streets_1 = filtered.Street_1
streets_2 = filtered.Street_2

# Cleaning some of the data from the dataset such that all streets have the same type of format 
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

# Merging the data to get a list of all of the streets of all of the crimes
streets = pd.concat([streets_1, streets_2], ignore_index=True)

# Remove duplicates to have only a list of unique streets of all the theft crimes
streets_unique = streets.unique()


# Saving this data to file such that it is not lost and the script does not have to be run fully to get this data
# streets_unique_df = pd.DataFrame(streets_unique) 
# writer = pd.ExcelWriter(r'C:\Users\Brenden Morton\Desktop\UCF\Fall2020\Network Optimization\Final Project\modifed_streets_cleaned.xlsx')
# streets_unique_df.to_excel(writer)
# writer.save()

# Read the cleaned data from the csv file
streets_unique_df = pd.read_excel('modifed_streets_cleaned.xlsx', index_col=0)

# Converting the datafram back to a list
streets_unique_df.columns = ['streets']        
streets_unique = list(streets_unique_df.streets)

# Initializing a tuple for storing the mapping of each street to a specific interger
# This is to be used in the construction of the adjacency matrix
street_mapping = {}

# Constructing the tuple of the form {street: i}, where i is an integer
for street,i in zip(streets_unique, range(len(streets_unique))):
    street_mapping.update({street:i})
 
# Initializing the matrix to all zeros
all_theft_matrix = np.zeros((len(streets_unique),len(streets_unique)))

# Resetting the index such that the first element starts at 0
streets_1.reset_index(drop=True, inplace=True)
streets_2.reset_index(drop=True, inplace=True)

# Constructing the symmetric adjacency matrix
for i,j in zip(streets_1, streets_2):
    all_theft_matrix[street_mapping[i]][street_mapping[j]] += 1
    all_theft_matrix[street_mapping[j]][street_mapping[i]] += 1
    

# Storing the matrix in a csv file so the script does not have to be run fully to get this data again
with open("cleaned_matrix.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(all_theft_matrix)
       

# Used for observing time of execution
print("--- %s seconds ---" % (time.time() - start_time))

# Resetting the indices of the data
filtered.reset_index(drop=True, inplace=True)

# Creating a tuple of the data
positions = {i:(filtered.Longitude[i],filtered.Latitude[i]) for i in range(len(filtered))}

# Converting the data into a numpy array
zatrix = nx.from_numpy_matrix(all_theft_matrix)


# Printing some charcateritics of the graph
print("Node of graph: ")
print(zatrix.nodes())
print("Edges of graph: ")
print(zatrix.edges())
print("info: ")
print(nx.info(zatrix, n=None))

# Setting up data to draw
ratrix = zatrix
ratrix = nx.to_numpy_matrix(ratrix)

# Running the Markov Clustering algorithm
res = mcl.run_mcl(ratrix)

# Obtaining the clusters of the data
clusters = mcl.get_clusters(res)

# Drawing the graph
mcl.drawing.draw_graph(ratrix, clusters, pos=positions,edge_color="red",node_size= 30, with_labels=False)
# mcl.drawing.draw_graph(ratrix, clusters,edge_color="red",node_size=0.5, with_labels=False)

# Storing the clusters of the data
with open("clusters.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(clusters) 


#-------------------------------------------------------------------------------------------------------------------    
# Network Centrality analysis with NetworkX    
    
# zatrix is the newtorkx graph

# Function that saves the data to a file with the specified path to the file
def write_to_file(data, file_path):
    # file_path = (r'C:\Users\Brenden Morton\Desktop\UCF\Fall2020\Network Optimization\Final Project' + f'\{path}')
    writer = pd.ExcelWriter(file_path)
    data = pd.DataFrame(data=data, index=[0])
    data.to_excel(writer)
    writer.save()
    
    

def get_street_from_key(key):
    return streets_unique[key]


# Getting the degree centrality of the data
deg_centrality = nx.degree_centrality(zatrix)
max_deg_centrality = max(deg_centrality, key=lambda k: deg_centrality[k])
print(f"Max degree centrality: {max_deg_centrality} ->" + get_street_from_key(max_deg_centrality))

# Getting the katz centrality of the data
katz_centrality = nx.katz_centrality(zatrix)
max_katz_centrality = max(katz_centrality, key=lambda k: katz_centrality[k])
print(f"Max katz centrality: {max_katz_centrality} ->" + get_street_from_key(max_katz_centrality))

# Getting the eirgenvector centrality of the data
eigen_centrality = nx.eigenvector_centrality(zatrix, max_iter=1000)
max_eigen_centrality = max(eigen_centrality, key=lambda k: eigen_centrality[k])
print(f"Max eigen vector centrality: {max_eigen_centrality} ->" + get_street_from_key(max_eigen_centrality))

# Getting the pageranks of the data    
pagerank = nx.pagerank(zatrix)
max_pageRank = max(pagerank, key=lambda k: pagerank[k])
print(f"Max pagerank: {max_pageRank} ->" + get_street_from_key(max_pageRank))

# Getting the hubs and authority data from HITS algorithm
hubs,authorities = nx.hits(zatrix, max_iter=500)
max_hub = max(hubs, key=lambda k: hubs[k])
max_authority = max(authorities, key=lambda k: authorities[k])
print(f"Max Hub: {max_hub} ->" + get_street_from_key(max_hub))
print(f"Max Authority: {max_authority} ->" + get_street_from_key(max_authority))


#  Creating a list of the number of occurrences for each street
occurrences = streets.value_counts()
occurrences.to_csv("occurrence_data.csv")

# List containing the degrees of each street
degree_list = [val for (node, val) in zatrix.degree()]
degree_list = pd.Series(degree_list)
degree_list = degree_list.sort_values(ascending=False)
degree_list.to_csv("degree_list.csv")


"""
top_list = [402,83,451,294]
top_list_cluster_mapping = {}

for tup in range(len(clusters)):
    for ele in tup:
        for top_ele in range(len(top_list)):
            if top_ele == ele:
                top_list_cluster_mapping.update({top_ele,tup})
"""          
    
    
# Uncomment to save network centrality data locally

#write_to_file(deg_centrality, r'C:\Users\Brenden Morton\Desktop\UCF\Fall2020\Network Optimization\Final Project\degree_centrality.xlsx')
#write_to_file(katz_centrality, r'C:\Users\Brenden Morton\Desktop\UCF\Fall2020\Network Optimization\Final Project\katz_centrality.xlsx')
#write_to_file(eigen_centrality, r'C:\Users\Brenden Morton\Desktop\UCF\Fall2020\Network Optimization\Final Project\eigen_centrality.xlsx')
#write_to_file(pagerank, r'C:\Users\Brenden Morton\Desktop\UCF\Fall2020\Network Optimization\Final Project\pagerank.xlsx')    
#write_to_file(hubs, r'C:\Users\Brenden Morton\Desktop\UCF\Fall2020\Network Optimization\Final Project\hubs.xlsx')
#write_to_file(authorities, r'C:\Users\Brenden Morton\Desktop\UCF\Fall2020\Network Optimization\Final Project\authorities.xlsx')
    

#-----------------------------------------------------------------------------------------------------------------------------------------------------
#Formatting for cytoscape

# export graph to cytoscape format
nx.write_gml(zatrix, "OrlandoCrimes_Theft.gml")

# export street names to each node
streets_unique_df = pd.DataFrame(streets_unique) 
#writer = pd.ExcelWriter(r'C:\Users\Brenden Morton\Desktop\UCF\Fall2020\Network Optimization\Final Project\modifed_streets_cleaned.xlsx')
#streets_unique_df.to_excel(writer)
#writer.save()


# Writing the node labels to file for cytoscape format

node_label_list = np.array(streets_unique_df)


# Formatting the data to be exported into Cytoscape
cyto_strings = []
for i,item in zip(range(len(node_label_list)), node_label_list ):
    item_indexed = item[0]
    cyto_strings.append(f'{i} = {item_indexed}')

# Saving the formatted data to a file    
with open("node_labels.txt", "w") as outfile:    
    outfile.write("\n".join(str(item) for item in cyto_strings))


