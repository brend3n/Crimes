"""
Program to determine the worst placees to park in the City of Orlando.

This is determined based off crime reports for Vehicle theft.


Author:  Brenden Morton
Date started: 11/22/2020
"""

import pandas as pd
import matplotlib.pyplot as plt


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

print(type(crime_filterd.Location.values[0]))


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
