# Pairwise Clustering
This repo contains a pairwise clustering library in pairwise_clustering.py that can be imported to perform pairwise clustering on two separate sequences of data given a distance or similarity function which takes in the objects in those sequences and returns a float. It is based on freelancing work I did for a client. The jupyter notebook pairwise_clustering_demo.ipynb shows an example of the pairwise_clustering function being used on demo data and performing very well.

## Demo Problem
In order to better understand some research, data has been gathered from the following public sources: PubMed, containing published scientific research, and clinicaltrials.gov, containing clinical trials.
The datasets must be merged together so that authors from the PubMed dataset are paired with investigators from the clinicaltrials.gov dataset. This can be done easily for a small subset of these researchers by pairing researchers with the same unique orcid value.
However, this value is not available for most researchers, so a different method must be used to pair them together. The Jupyter notebook shows analysis of the datasets, constructs a similarity score based on the analysis, and uses the pairwise_clustering function to get matches.

## Demo Data
The two datsets are represented in two csv files. The PubMed data is contained in author.csv and the clinicaltrials.gov data is contained in investigator.csv.  
Example instances from the author.csv:  
|first_name|middle_name|lastname|coauthor_lastnames|topics|cities|countries|orcid|
|---|---|---|---|---|---|---|---|
|a|NA|walker|["abou raya","helmii"]|["Geriatric Assessment","Case-Control Studies"]|[{"identifier":361058,"name":"Alexandria"}]|[{"identifier":357994,"name":"Egypt"}]|NA|
|stefan|NA|majumdar|["zaman","thurlimann","brauchli","pagani"]|["Tamoxifen","Antineoplastic Combined Chemotherapy Protocols","Aged"]|[{"identifier":2659811,"name":"Luzern"},{"identifier":2661552,"name":"Bern"}]|[{"identifier":2658434,"name":"Switzerland"}]|653524|

Example instances from the investigator.csv:  
|first_name|middle_name|lastname|coinvestigator_lastnames|topics|cities|countries|orcid|
|---|---|---|---|---|---|---|---|
|m|NA|li|["sherman"]|["Osteoporosis"]|[{"identifier":null,"name":null}]|[{"identifier":null,"name":null}]|NA|
|s|NA|majumdar|["aarau","di bergamo","bern"]|["Tamoxifen","Osteoporosis"]|[{"identifier":2659811,"name":"Luzern"},{"identifier":2661552,"name":"Bern"}]|[{"identifier":2658434,"name":"Switzerland"}]|653524|