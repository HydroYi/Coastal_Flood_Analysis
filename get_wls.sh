#!/bin/bash



function get_wls(){ # give station id and then output file name ...............set dates \/ here   and  here \/
wget -O $2 "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date=20200401 00:00&end_date=20200501 00:00&station=$1&product=water_level&datum=igld&units=metric&time_zone=gmt&interval=h&application=NOAA-GLERL&format=csv"
}


get_wls 9087057 milwaukee.csv 
get_wls 9087031 holland.csv
get_wls 9075014 harbor_beach.csv
get_wls 9075065 alpena.csv
get_wls 9087023 ludington.csv



# if you had a bunch of stations to get
#ids=( $(awk 'NR>1{print $1}' stations.txt) )
#for id in ${ids[@]}; 
#	do echo getting $id; 
#done
