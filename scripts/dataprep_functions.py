## !!! Difference from previous notebooks: community labels (which are calculated from several runs, so they are consensus communities) are named 'S' INSTEAD OF 'S_cons' as it is shorter


# 0.) IMPORT PACKAGES

import pandas as pd
import geopandas as gpd
import json
import networkx as nx

import numpy as np
import scipy

from copy import deepcopy

from itertools import product # as iter_product

from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score # Adjusted Mutual Information (adjusted against chance)
from sklearn.metrics.cluster import adjusted_rand_score # Adjusted Rand Index

from statistics import median

# KERDES octave-ot hogy hasznaljam, a lentebbiek mik? máskor hogyan kellene ezt megcsinálnom??
from oct2py import octave
_ = octave.addpath('/home/barcsab/projects/urban_communities/scripts')
_ = octave.addpath('/home/ubuntu/GenLouvain/')
_ = octave.addpath('/home/ubuntu/GenLouvain/private/')

########### MINDEN ÖSSZEÖNTVE, LEHET VAN BENNE FELESLEGES - TODO ATNEZNI

import numpy as np
import seaborn as sns

import csv

import matplotlib.cm as cm
import matplotlib.pyplot as plt

import community as community_louvain

# from modularity_maximization.utils import get_modularity

from itertools import product
import networkx.algorithms.community as nx_comm
from scipy.spatial.distance import pdist, squareform

from shapely.geometry import Point, LineString
from geopandas import GeoDataFrame

import math
from time import time

import matplotlib.lines as lines


# 1.) IMPORT DATA

# 1.1.) 3 networks

# 1.1.1.) aggregated to census tract level
mobility = pd.read_csv("../data/usageousers_city_mobility_CT_networks.rpt.gz") # basis of position and node importance calculations
follow_hh = pd.read_csv("../data/usageousers_city_follower_CT_HH_networks.rpt.gz")
follow_hh = follow_hh.rename(columns={"tract_home.1": "tract_home_1"})

# 1.1.2.) DROP THE DUPLICATE TRACTS FROM THE CITY WHERE IT HAS THE LOWER DEGREE
# There are same geoids that corresponds to several cities which is an error. We choose which city it should belong to based on degree.)

# how many cities do geoids belong to in the mobility dataframe?
temp = mobility\
        .melt(['cbsacode','cnt'])\
        .groupby(['cbsacode','value'])\
        .sum()['cnt']\
        .sort_values(ascending=False)\
        .reset_index()\
        .groupby('value')\
        .count()['cbsacode']\
        .sort_values()
# degree of geoids within cities in the mobility dataframe?
temp2 = mobility\
    .melt(['cbsacode','cnt'])\
    .groupby(['cbsacode','value'])\
    .sum()['cnt']\
    .sort_values(ascending=False)\
    .reset_index()
# selecting geoids that are in two cities, and getting the lower degree variant for later exclusion
mob_to_exclude = temp2\
    .set_index('value').loc[temp[temp>1].index]\
    .reset_index()\
    .drop_duplicates(subset=['value'], keep='last').iloc[:,[1,0]]

mob_to_exclude_l = list(zip(mob_to_exclude['cbsacode'], mob_to_exclude['value']))


# how many cities do geoids belong to in the follower dataframe?
temp = follow_hh\
        .melt(['cbsacode','cnt'])\
        .groupby(['cbsacode','value'])\
        .sum()['cnt']\
        .sort_values(ascending=False)\
        .reset_index()\
        .groupby('value')\
        .count()['cbsacode']\
        .sort_values()
# degree of geoids within cities in the follower dataframe?
temp2 = follow_hh\
    .melt(['cbsacode','cnt'])\
    .groupby(['cbsacode','value'])\
    .sum()['cnt']\
    .sort_values(ascending=False)\
    .reset_index()
# selecting geoids that are in two cities, and getting the lower degree variant for later exclusion
fol_to_exclude = temp2\
    .set_index('value').loc[temp[temp>1].index]\
    .reset_index()\
    .drop_duplicates(subset=['value'], keep='last').iloc[:,[1,0]]

fol_to_exclude_l = list(zip(fol_to_exclude['cbsacode'], fol_to_exclude['value']))

# drop from mobility_df

mobility['tuple_1'] = list(zip(mobility['cbsacode'], mobility['tract_home']))
mobility['tuple_2'] = list(zip(mobility['cbsacode'], mobility['tract_work']))

mobility = mobility[-mobility['tuple_1'].isin(mob_to_exclude_l)]
mobility = mobility[-mobility['tuple_2'].isin(mob_to_exclude_l)]

# drop from follow_hh 

follow_hh['tuple_1'] = list(zip(follow_hh['cbsacode'], follow_hh['tract_home']))
follow_hh['tuple_2'] = list(zip(follow_hh['cbsacode'], follow_hh['tract_home_1']))

follow_hh = follow_hh[-follow_hh['tuple_1'].isin(fol_to_exclude_l)]
follow_hh = follow_hh[-follow_hh['tuple_2'].isin(fol_to_exclude_l)]
follow_hh = follow_hh[-follow_hh['tuple_1'].isin(mob_to_exclude_l)]
follow_hh = follow_hh[-follow_hh['tuple_2'].isin(mob_to_exclude_l)]

# 1.1.3.) census tract name --> cbsacode
cbsacode = pd.read_csv("../data/cbsacode_shortname_tracts.csv",sep=";", index_col=0)
cbsacode['clean_name'] = cbsacode["short_name"].map(lambda s: s.split("/")[0].replace(' ','_').replace('.','').lower())

# drop
cbsacode['tuple'] = list(zip(cbsacode['cbsacode'], cbsacode['geoid']))
cbsacode = cbsacode[-cbsacode['tuple'].isin(fol_to_exclude_l)]
cbsacode = cbsacode[-cbsacode['tuple'].isin(mob_to_exclude_l)]


# 1.2.) GeoData

# 1.2.1.) reading geojson data, converting it to geopandas dataframe
tract_geoms = gpd.GeoDataFrame.from_features(
    [json.loads(e.strip('\n')) for e in open('../data/censustract_geoms_top50.geojson').readlines()]
)

# 1.2.2.) Cartesian coordinate projection of tract centroids
tract_geoms['centroid'] = tract_geoms['geometry'].centroid
tract_center_dict = tract_geoms\
    .set_geometry('centroid',crs={'init':'epsg:4326'})\
    .to_crs({'init':'epsg:3785'})\
    .set_index('full_geoid')['centroid'].map(lambda p: p.coords[0]).to_dict()


# 1.3.) Census - Demographic data at tract level
census_1 = pd.read_csv("../data/censusdata_top50_2012.csv")
census_2 = pd.read_csv("../data/censusdata_top50_2017.csv")

# 2.) FUNCTIONS

# 2.1.) data preparation - create tract geoids
def create_geoid(row):
    state = str(int(row["state"])).zfill(2)
    county = str(int(row["county"])).zfill(3)
    tract = str(int(row["tract"])).zfill(6)
    return "14000US" +state+county+tract


# 2.2.) auxiliary function

###### START HERE, CHECK

# copied from Spatial_modularity_pooled notebook
def create_graph(city, g_type): ### ATIRTAM KICSIT, ELLENORIZNI
    """
    For a given city name, it generates a mobility or follower (home-home) graph.
    
    e.g. g = create_graph("Boston")
    
    It uses the previously loaded `mobility` and `follow_hh` pandas.DataFrames, in which
    the edges are listed for every city.
    
    Parameters:
    -----------
    city : str
        name of the city, see cbsacode dataframe -> clean_name
        
    g_type : str
        either "mob" as mobility or "fol_hh" as follow_hh 
        selects the type of graph to return
        
    Returns:
    --------
    
    g : networkx.Graph
        weighted undirected graph based on city name and g_type (e.g. follow_hh graph of Boston)
        
    """
    # city cbsacode based on name
    city_code = cbsacode[cbsacode.clean_name == city].iloc[0].cbsacode
    
    # select graph type
    if g_type == "mob":
        # filtering large dataframes for the given city code
        mob_df = deepcopy(mobility[(mobility["cbsacode"] == city_code)&(mobility["tract_home"]!=mobility["tract_work"])])

        # create graphs
        # create empty graphs
        g_mob = nx.DiGraph() # mobility graph - weights are counts

        # fill in the networks with data
        mob_df['w_edges'] = list(zip(mob_df.tract_home,mob_df.tract_work,mob_df.cnt))
        g_mob.add_weighted_edges_from(mob_df["w_edges"], weight='cnt')

        # ineffective and slow!
        for e in g_mob.edges():
            r = (e[1],e[0])

            if r in g_mob.edges():
                c1 = g_mob.edges[e]['cnt']
                c2 = g_mob.edges[r]['cnt']

                g_mob.edges[e]['cnt'] = c1 + c2
                g_mob.edges[r]['cnt'] = c1 + c2

        # then let's convert the mobility graph to udirected
        g_mob = g_mob.to_undirected()

        g = g_mob
        
    elif g_type == "fol_hh":            
        # filtering large dataframes for the given city code
        fol_hh_df = deepcopy(follow_hh[(follow_hh["cbsacode"] == city_code)&(follow_hh["tract_home"]!=follow_hh["tract_home_1"])])

        # create graphs
        # create empty graphs
        g_fol_hh = nx.Graph() # follow home-home graph - weights are counts

        # this is an undirected graph already in the dataframe
        fol_hh_df['w_edges'] = list(zip(fol_hh_df.tract_home,fol_hh_df.tract_home_1,fol_hh_df.cnt))
        g_fol_hh.add_weighted_edges_from(fol_hh_df["w_edges"], weight='cnt')
        
        g = g_fol_hh
        
    # check data - if all nodes of the graph are in the tract_geom dataframe
    # e.g. in create_graph()
    # if someone's not there, that is data error, print the tract_id, and leave the node out of the graph G
    # only after this should we calculate the Expert input data    
    while not set(g.nodes).issubset(set(tract_geoms.full_geoid)): # KERDES: ezt hogyan ellenőrizzem le?
        print('DATA ERROR. Node do(es) not have corresponding geodata, so dropped.')
        print('Dropped node(s):')
        ####nodes_to_drop = set(g.nodes).difference(set(tract_geoms.full_geoid))
        ####g.remove_nodes_from(nodes_to_drop)
    return g

#### END HERE CHECK


# 2.3.) DEGREE AND DEGREECENTRALITY
# TODO KIPROBALNI

def degree_degreecent_dict(city, g_type):
    # make the graph
    G = create_graph(city, g_type) # kell ez a sor??
    # calculate degree and degree centrality for each node
    degree_dict = dict(nx.degree(G, weight="cnt"))
    degreecent_values = np.array(list(degree_dict.values())) / np.array(list(degree_dict.values())).sum()
    degreecent_dict = dict(zip(degree_dict.keys(), degreecent_values)) ## ezt így lehet?? KERDES: degree_dict.keys() kell ide?
    return degree_dict, degreecent_dict

# 2.4.) Modularity calculations
## Ms/Mgn - deterrence function nem, mik ezek???? KERDES?
# calculations based on Newman-Girvan modularity (standard) and Expert modularity (spatially corrected: null model account for the fact that the closer two places to each other the more connection is expected between them)


# 2.4.1.) Modularity calculation - matrices for clustering for both algorithm_type-s (1 iteration)
### START HERE
# (((copied from Spatial_community_pooled notebook)))
def SpaMod(A,D,N,binnumber): # binnumber instead of b = binsize
    """
    Function that calculates the matrix for the clustering 
    based on spatial null model a la Expert.
    
    Parameters:
    -----------
    
    A : scipy.sparse.csr.csr_matrix
        adjacency matrix
    D : numpy.ndarray
        Distance matrix between the nodes
    N : numpy.matrix
        a measure of the importance of a node
        the number of users living(home-location) in the given tract
    binnumber : int
        number of distance bins (used in the estimation of the deterrence function)
    Returns:
    --------
    
    KERDES - ellenorizni
    ModularitySpa : 
    ModularityGN :
    """
     
    # felesleges?? KERDES -- symmetrised matrix (doesn't change the outcome of community detection (arXiv:0812.1770))
    A = A + A.T ### KERDES KELL-e?? TODO ATGONDOLNI? ILLETVE LE KELL-e osztani 2-vel   / 2     
    b = D.max()/(binnumber-1) # MODIFIED
    
    # deterrence function
    det, detbins = np.histogram(
        D.flatten(),
        range = (0, np.ceil(D.max()/b)*b), # JAVITAS
        weights = np.array(A.todense()).flatten(), 
        bins=int(np.ceil(D.max()/b))
    )
    normadet, _ = np.histogram(
        D.flatten(), 
        range = (0, np.ceil(D.max()/b)*b),
        weights = np.array(N*N.T).flatten(), 
        bins=int(np.ceil(D.max()/b))
    )
    det = det / normadet
    det[np.isnan(det)] = 0
    
        
    # copmutation of the randomised correlations (preserving space), spatial
    # null-model
    nullmodelSpa = det[np.digitize(D,detbins,right=True)-1]
    
    # the modularity matrix for the spatial null-model
    ModularitySpa=A-np.multiply(N*N.T, nullmodelSpa*A.sum())/(np.multiply(N*N.T,nullmodelSpa).sum())
    szamlalo = np.multiply(N*N.T, nullmodelSpa*A.sum())
    nevezo = np.multiply(N*N.T,nullmodelSpa).sum()
    
    # the modularity matrix for the GN null-model
    degree = degree = A.sum(axis=0) # JAVITVA np.squeeze(np.asarray(A.sum(axis=0))) # degree or strength of nodes || asarry for further usage
    nullmodelGN = degree.T*degree/degree.sum() # Newman-Girvan null-model
    ModularityGN = A - nullmodelGN
       
    return ModularitySpa, ModularityGN # KERDES UTOLAG: EZ AZT ADJA MEG, HOGY ADOTT KÉT CSÚCS KÖZÖTT MENNYIVEL TÖBB ÉL VAN A VÁRTNÁL??
# KERDES FOLYT.: EGY KLASZTEREZÉS MODULARITÁSÁT ÚGY KAPOM MEG EBBŐL, HOGY ÖSSZEADOM AZON ELEMEIT, AMIK MINDKÉT CSÚCSA UGYANAHHOZ A COMMUNITYHEZ TARTOZIK, MAJD EOLSZTOM 2M-MEL (2*m = 2-szer total weight of the network = 2*szum(A)/2 = szum(A)

### END HERE CHECK



# CHECK START HERE - from Spatial_community_pooled
# 2.4.2.
# Consensus clustering
def consen(city, algorithm_type, g_type):
    """
    Function that does the consensus clustering based on the results
    of multiple runs of previous algorithms.

    -- It makes a weighted graph with the same nodes (tracts) as the origianl clustering
    and edge weights are the number of iteration when the two nodes were clustered to the same community according to the input dataset.
    The consensus clustering is the clustering made on this new network
    using ordinary community detection calculating modularity a la Newman-Girvan. ??kerdes ez newman-girvan, ugye?  
    
    Parameters:
    -----------
    
    city : str
        cityname to run the consensus clustering for (see cbsacode.clean_name)
    algorithm_type: str
        either "ms" or "mgn" 
        selects the clustering algoritm type: spatail null model (modularity calculation - a la Expert) or ordinary null model (modularity calcualted a la Girvan-Newman)
    g_type : str
        either "mobility" or "follow_hh"
        selects the type of graph: commuting between home and workplace OR mutual?? followership socail ties using homelocations
        
    Returns:
    --------
    
    s_louv : dict
        tract_geoid -> partition label (int)
    """
    
    # Reading in necessary data
    csv = '../data/com_detect_iters_' + city + '_' + algorithm_type + '_' + g_type + '.csv'

    # results of multiple iterations from previous runs
    iters = pd.read_csv(csv)
    iters = iters.set_index('geoid')
    iters['clusts'] = [np.array(l) for l in iters.values.tolist()]

    # create all possible node pairs
    geoid_pairs = list(product(list(iters.index), list(iters.index)))
    consen_df = pd.DataFrame(geoid_pairs, columns=['geoid_1','geoid_2'])

    # remove selfloops
    consen_df = consen_df[consen_df.geoid_1!=consen_df.geoid_2]
    

    # joining iteration results (community label) as lists to nodes (- KERDES: jo, utolag kommentaltam at!!!) (It is needed to b done for geoid_1 and geoid_2 as a node can be on either end of the edge???)
    # both elements of the node(=tract) pairs
    consen_df = pd.merge(consen_df, iters['clusts'], left_on = 'geoid_1', right_on = 'geoid')
    consen_df = pd.merge(consen_df, iters['clusts'], left_on = 'geoid_2', right_on = 'geoid')
    consen_df = consen_df.rename(columns = {'clusts_x': 'clusts_1', 'clusts_y': 'clusts_2'})
  

    # Edge weights <-- no. of iterations when the two nodes (tracts) defining the edge are clustered to the same community
    ## how many times are the two nodes (=tracts) (geoid_1 and geoid_2) clustered to the same community?
    ## --> weights of a graph on which clustering gives the consensus clustering
    
    same_com = np.array(consen_df['clusts_2'].tolist()) == np.array(consen_df['clusts_1'].tolist())
    del consen_df['clusts_1'], consen_df['clusts_2']
    consen_df['w'] = same_com.sum(axis=1)
    del same_com

    # graph for consensus clustering
    # ((create a graph from the consen_df edgelist (nodes (tracts) are same as in the original network, but the edge weights are the number of iterations when they are given the same community label))
    g_cons = nx.Graph()
    g_cons.add_nodes_from(consen_df['geoid_1'])
    g_cons.add_nodes_from(consen_df['geoid_2']) # KERDESES - ez lehet, hogy mr felesleges
    
    #remove missing edges from edgelist (delete edges with 0 weight)
    consen_df = deepcopy(consen_df[consen_df['w']!=0])
    
    g_cons.add_weighted_edges_from(consen_df[['geoid_1','geoid_2','w']].values, weight='w')
    
    del consen_df, iters

    # Louvain community detection 
    s_louv = community_louvain.best_partition(g_cons, weight='w')

    return s_louv

# END HERE CHECK__________________________________



# 2.4.3.) Modularity contribution of communities to the overall modularity of networks

### TODO: MODULARITY CONTRIBUTION NEM MÉRETFÜGGŐ?

### START

def community_modularity(city, g_type, tract_df):
    """
    Calculates how much a certain community contributes to the overall modularity.
    
    Uses the create_function and SpaMod functions.
    
    Parameters:
    -----------
    city : str
        as in cbsacode.clean_name ex: 'salt_lake_city'
    
    g_type : str
        'mob' or 'fol_hh' as mobility (commuting) or follower home-home (social ties)
        
    Returns:
    ---------
    
    pandas.DataFrame with measure of how much a given community contributed to the overall modularity of the given network's
    partitioning with for the two algorithms.

    """
    
    G = create_graph(city, g_type) # corresponding weighted undirected graph
    
    # Dataprep for Expert algorithm
    A = nx.adjacency_matrix(G)
    coords = np.array([tract_center_dict[n] for n in G.nodes()])
    d = pdist(coords)
    D = squareform(pdist(coords))
    tract_outdeg_mob = mobility.groupby('tract_home')[['cnt']].sum()
    N = np.matrix([tract_outdeg_mob.loc[k].iloc[0] for k in G.nodes()]).T

    Ms,Mgn = SpaMod(A,D,N,200)
    
    
    mod_df = pd.DataFrame(columns=['city', 'algorithm_type', 'g_type', 'S', 'modularity_S'])

    for (modularity, algorithm_type) in [(Ms, 'ms'), (Mgn, 'mgn')]:
        S_notsorted = deepcopy(tract_df[\
            (tract_df['city'] == city) & \
            (tract_df['algorithm_type'] == algorithm_type) & \
            (tract_df['g_type'] == g_type)\
        ].set_index('geoid')['S'])
        S = [S_notsorted.loc[k] for k in G.nodes() if k in S_notsorted] # KERDES: kell ez?

        for s in list(set(S)):
            com_vector = (np.array(S) == s).astype('int')
            # clusters to matrix format
            # matrix for each cluster (in which 1s are for node pairs which are in the given community all others are 0s)
            # same community matrix (stored as array) - 0 if either of the nodes is not in community, 1 if both are in community
            same_com = np.outer(com_vector, com_vector)
      
            # add up how much more edges are present between community nodes then expected by nullmodel in the given community
            mod = (np.multiply(np.array(modularity), same_com).sum()) / (2 * A.sum())
            
            mod_df = mod_df.append({'city': city, 'algorithm_type' : algorithm_type, 'g_type' : g_type, 'S' : s, 'modularity_S' : mod}, ignore_index=True)
    
    return mod_df


### END CHECK


# 3.) CALCULATIONS USING THE FUNCTIONS AND DATA ABOVE

# 3.1.) MAKE A SINGLE census_df (merge, add suffix and geoid)
def merge_census(census_1, census_2): # vagy globalba a hasa helyett?

    census_1['geoid'] = census_1.apply(create_geoid,axis=1)
    census_2['geoid'] = census_2.apply(create_geoid,axis=1)

    census_1 = census_1.add_suffix('_1')
    census_2 = census_2.add_suffix('_2')

    census_df = pd.merge(census_1, census_2, left_on = 'geoid_1', right_on = 'geoid_2')
    census_df = census_df.drop(columns = ['geoid_2'])
    census_df = census_df.rename(columns = {'geoid_1': 'geoid'})
    return census_df

# 3.2.) Make list of city names
city_l = list(cbsacode['clean_name'].unique())

# 3.3.) RUN AND STORE RESULTS OF 20 ITERATIONS OF COMMUNITY DETECTIONS

# (copied from spatial_community_pooled.ipynb, some comments deleted)
def community_detection_iters(): 

    tract_outdeg_mob = mobility.groupby('tract_home')[['cnt']].sum()

    for city in city_l:
        for g_type in ['mob','fol_hh']:
            print(f"Calculating iterations for {city}, graph type {g_type}...")
            G = create_graph(city, g_type) # corresponding weighted undirected graph
            
            # inS_notsorteddex conversion dicts
            # geoid -> integer 0-... N-1
            # int -> geoid
            index_geoid_dict = dict(list(enumerate(G.nodes)))
            geoid_index_dict = dict(zip(list(index_geoid_dict.values()), list(index_geoid_dict.keys())))

            
            # Dataprep for Expert algorithm
            A = nx.adjacency_matrix(G)
            coords = np.array([tract_center_dict[n] for n in G.nodes()])
            d = pdist(coords)
            D = squareform(pdist(coords))
            
            # importance - number of user home in each tract
            # check if all nodes in the follow_hh graph have an importance!
            # otherwise, the N... line is going to throw an error
            if not set(G.nodes).issubset(set(tract_outdeg_mob.reset_index().tract_home)): # test if the node is in any city ((KERDES : adott városra teszteljem?))
                print('Error. Node(s) without importance value(s) They are dropped.')
                missing_nodes = list(set(G.nodes)-set(tract_outdeg_mob.reset_index().tract_home))
                for node in missing_nodes:
                    #((# KERDES - ezt ki is dobjam??))
                    G.remove_node(node)    
            N = np.matrix([tract_outdeg_mob.loc[k].iloc[0] for k in G.nodes()]).T
            
            
            # Calculate clusterings for the given graph and write the outcome of runs to csvs
            S_ms_df = pd.DataFrame()
            S_mgn_df = pd.DataFrame()
            for _ in range(10):
                print(_, end=", ")
                # TODO Eszter!!!! sometimes it gives an error in the first line
                Ms,Mgn = SpaMod(A,D,N,200) #((## KERDES what should be the number of bins? 100?))
                S_ms,Q_ms,n_it_ms = octave.iterated_genlouvain(Ms, nout=3)
                S_ms_df[len(S_ms_df.columns)] = S_ms.T[0]
                S_mgn,Q_mgn,n_it_mgn = octave.iterated_genlouvain(Mgn, nout=3)
                S_mgn_df[len(S_mgn_df.columns)] = S_mgn.T[0]

                for (algorithm_type, df) in [('ms',S_ms_df),('mgn',S_mgn_df)]:
                    df['geoid'] = df.index.map(index_geoid_dict)
                    df = df.set_index('geoid')
                    csv_name = 'com_detect_iters_' + city + '_' + algorithm_type + '_' + g_type + '.csv'
                    df.to_csv('../data/'+ csv_name)
            print("Done.\n")


# 3.4.) CALCULATE CONSENSUS CLUSTERING
city_gtype_alg_combs = product(city_l, ['mob','fol_hh'], ['ms','mgn'])

def calc_consen():
    all_consensus_df = pd.DataFrame()
    for city, g_type, algorithm_type in city_gtype_alg_combs:
            print(city)
            print(g_type)
            print(algorithm_type)
            # storing iteration results, empty dataframe for nodes
            consensus_df = pd.DataFrame()
            
            S = consen(city, algorithm_type, g_type)
            consensus_df['S'] = S.values()
            consensus_df['city'] = city
            consensus_df['algorithm_type'] = algorithm_type
            consensus_df['g_type'] = g_type
            consensus_df['geoid'] = S.keys()
            consensus_df = consensus_df.set_index('geoid')
            
            all_consensus_df = pd.concat([all_consensus_df, consensus_df])

    all_consensus_df.to_csv('../data/consensus_clust.csv')



# 3.5.) CREATE tract_df AND ADD DEGREE AND DEGREECENTRALITY TO THE DATASET
def create_tract_df_with_degree():
    
    tract_df = pd.read_csv('../data/consensus_clust.csv')

    degree_df = pd.DataFrame()
    degreecent_df = pd.DataFrame()

    graph_combs = product(city_l, ['mob','fol_hh'])
    for city, g_type in graph_combs:
        degree_dict, degreecent_dict = degree_degreecent_dict(city, g_type)

        # hibas? degree_df_iter = {degree_dict, orient = 'index', columns = ['degree']}
        # hibas? degreecent_df_iter = {degreecent_dict, orient = 'index', columns = ['degreecent']}
        degree_df_iter = pd.DataFrame.from_dict(degree_dict, orient='index', columns = ['degree']).reset_index()
        degree_df_iter = degree_df_iter.rename(columns = {'index' : 'geoid'})       

        degreecent_df_iter = pd.DataFrame.from_dict(degreecent_dict, orient = 'index', columns = ['degreecent']).reset_index()
        degreecent_df_iter = degreecent_df_iter.rename(columns = {'index' : 'geoid'})

        degree_df_iter['g_type'] = g_type
        degreecent_df_iter['g_type'] = g_type

        degree_df = pd.concat([degree_df, degree_df_iter])
        degreecent_df = pd.concat([degreecent_df, degreecent_df_iter])

    tract_df = pd.merge(tract_df, degree_df, how = 'left', on = ['g_type', 'geoid'])
    tract_df = pd.merge(tract_df, degreecent_df, how = 'left', on = ['g_type', 'geoid'])
       
    return tract_df

# 3.6.) community_df

# 3.6.1.) CREATE community_df
def create_community_df(tract_df):
    community_df = tract_df.groupby(['city', 'algorithm_type', 'g_type', 'S'])['degree', 'degreecent'].mean().reset_index()
    community_df = community_df.rename(columns = {'degree' : 'degree_avg', 'degreecent' : 'degreecent_avg'})
    return tract_df, community_df

# 3.6.2.) COMMUNITY MODAULARITY - ellenőrizni kellene HIBAS KERDESES 0601

def calc_community_modularity(community_df, tract_df):
    """
    Calculate how much a community contributes to the overall network modularity and store it in community_df.
    """
    com_mod_df = pd.DataFrame()

    graph_combs = product(city_l, ['mob','fol_hh'])  
      
    for city, g_type in graph_combs:
        df = community_modularity(city, g_type, tract_df)
        com_mod_df = pd.concat([com_mod_df, df])
    community_df = pd.merge(community_df, com_mod_df, how = 'left', on = ['city', 'algorithm_type', 'g_type', 'S']) ## TODO ELLENORZES megnézni, hogy hány sora marad, ha nem left, hanem inner, jó-e ez a merge
    return community_df, com_mod_df
    
# 3.7.) network_df

# 3.7.1.) MODULARITY CALCULATIONS
# 3.7.1.1.) CREATE network_df WITH OVERALL NETWORK MODULARITY
def calc_modularity(com_mod_df):
    """
    Calculate overall network modularity and store it in network_df.
    """
    network_df = deepcopy(com_mod_df.groupby(['city', 'algorithm_type', 'g_type'])[['modularity_S']].sum().reset_index())
    network_df = network_df.rename(columns = {'modularity_S' : 'modularity'})
    return network_df

# 3.7.1.2.) CALCULATE MODULARITY CONTRIBUTION AND ADD IT TO community_df
# modularity contribution = mod_community/mod_network
def calc_modularity_contribution(community_df, network_df):
    """
    Calculate the proportion of the network modularity originating from the community and store it in community_df.
    Modularity contribution = Modularity of the network / "modularity" of the community
    """

    community_df = pd.merge(community_df, network_df, how = 'left', on = ['city', 'algorithm_type', 'g_type'])
    community_df['mod_S_p'] = community_df['modularity_S'] / community_df['modularity']
    return community_df

# 3.7.1.3.) TODO talán később -- Community quality - RANK BASED ON mod_S_p 
# IDERAKNI - félig van csak kész


# 3.7.2.) ADD NUMBER OF TRACT IN NETWORK, IN CITY AND IN COMMUNITY AND ADD IT TO network_df AND community_df RESPECTIVELY
def calc_tract_number(tract_df, community_df, network_df, cbsacode): # TODO 0608 esetleg ide rakni a fentebbi tract számítást is!!
    """
    Calculate number of tracts in network, in city and in community and store it in network_df and community_df respectively.
    """
    
    # number of tracts or number of users in it?? KERDES
    # sum 'ones' or sum 'cnt' ?
    # number of tracts in community
    tract_df['ones'] = 1
    community_cnt_df = tract_df.groupby(['city', 'algorithm_type', 'g_type', 'S'])['ones'].sum().reset_index()
    community_cnt_df = community_cnt_df.rename(columns = {'ones' : 'tract_sum'})
    community_df = pd.merge(community_df, community_cnt_df, how = 'left', on = ['city', 'algorithm_type', 'g_type', 'S']) # TODO LATER: ellenőrizni, hogyha inner, akkor elveszik-e belőle sor, mert nem szabadna

    # number of tracts in network
    network_df = pd.merge(network_df, community_cnt_df.groupby(['city', 'algorithm_type', 'g_type'])[['tract_sum']].sum().reset_index(), how = 'left', on = ['city', 'algorithm_type', 'g_type']) # TODO ellenorizni, hogyha inner lenne left helyett, akkor elveszne-e sor, mert nem szabadna

    # number of tracts in city
    city_cnt_df = deepcopy(cbsacode.groupby(['clean_name'])['short_name'].count().reset_index())
    city_cnt_df = city_cnt_df.rename(columns = {'short_name' : 'tract_city_sum'})
    network_df = pd.merge(network_df, city_cnt_df, how = 'left', left_on = ['city'], right_on = ['clean_name']) # TODO ellenőrizni, h left helyett inner mergenél kiesne-e sor, nem szabadna
    network_df = network_df.drop(columns = ['clean_name'])

    return community_df, network_df



# 3.7.3.) ADD DENSITY TO network_df
def calc_network_density(network_df):
    """
    Calculate network density and store it in network_df.
    """
    network_df['density'] = network_df.apply(lambda row: nx.density(create_graph(row['city'], row['g_type'])),axis=1)
    return network_df

# 3.7.4.) SIMILARITY BETWEEN COUNTIES AND COMMUNITIES
# Normalized Mutual Information
## How similar is the clustering to the county system? correlation, symmetric function
##  --> the higher the index the more similar the clustering to county system

def calc_community_county_similarity(tract_df, network_df):
    """
    Calculate Normalized Mutual Information, Adjusted Normalized Mutual Information and Adjusted Rand Index
    which describe the similarity of county-community system, and store it in network_df.
    The higher the index the more similar the two groupings are.
    """
    # get the county for all tracts
    tract_df['county'] = tract_df['geoid'].map(lambda i: i[9:12])

    # empthy dataframe
    nmi_all_df = pd.DataFrame(columns=['city','g_type','algorithm_type','nmi_to_counties','adj_nmi_to_counties','adj_rand_to_counties'])

    # calculate every index for every community structure
    for city, g_type, algorithm_type in city_gtype_alg_combs:

        nmi_df = deepcopy(tract_df[(tract_df['city'] == city) & (tract_df['g_type'] == g_type) & (tract_df['algorithm_type'] == algorithm_type)])
        nmi = normalized_mutual_info_score(nmi_df['county'], nmi_df['S'], average_method='arithmetic')
        a_nmi = adjusted_mutual_info_score(nmi_df['county'], nmi_df['S'], average_method='arithmetic')
        a_rand = adjusted_rand_score(nmi_df['county'], nmi_df['S'])
        nmi_all_df = nmi_all_df.append({'city': city, 'g_type' : g_type, 'algorithm_type' : algorithm_type, 'nmi_to_counties' : nmi, 'adj_nmi_to_counties' : a_nmi, 'adj_rand_to_counties' : a_rand}, ignore_index=True)  

    # merge it to network_df
    network_df = pd.merge(network_df, nmi_all_df, how = 'left', on = ['city', 'g_type', 'algorithm_type'])
    return network_df



# 4.) CENSUS CALCULATIONS - KERDES: FOR CITY VS FOR NETWORK!!!!!!

# KERDES - TISZTITSAM ennel a pontnal AZ ADATOT???

# NINCS PRÓBÁLVA ettől lejjebb
# 4.1.) DROP UNNECESARRY COLUMNS AND RENAME - KERDES: ez igy ok?
def clean_census_to_tract_df(tract_df, census_df):
    census_df = census_df.drop(columns = ['state_1', 'county_1', 'tract_1', 'population_error_1',
       'education_bachelor_error_1', 'education_total_1', 'education_total_error_1',
       'income_error_1', 'race_total_1', 'race_total_error_1',
       'white_error_1', 'black_error_1', 'native_error_1', 'asian_error_1',
       'state_2', 'county_2', 'tract_2', 'population_error_2',
       'education_bachelor_error_2', 'education_total_2', 'education_total_error_2',
       'income_error_2', 'race_total_2', 'race_total_error_2',
       'white_error_2', 'black_error_2', 'native_error_2', 'asian_error_2'])

    census_df = census_df.rename(columns= {'education_bachelor_1' : 'educ_ba_1', 'education_bachelor_2' : 'educ_ba_2'})

# 4.2.) MERGE CENSUS DATA TO tract_df
    tract_df = pd.merge(tract_df, census_df, how = 'left', on = ['geoid'])
    return tract_df, census_df


### elnevezes S_CONS HELYETT S, S HELYETT S_NONCONS

# 4.3.) ADD SUM and STD OF CENSUS DATA TO network_df AND community_df
def calc_sum_and_std(tract_df, community_df, network_df):
    """
    Calculate the sum and standard error of tract level socioeconomic data (census) for community and network level
    and store it in community_df and network_df.
    """

    com_sum_df = tract_df.groupby(['city','algorithm_type','g_type','S'])['population_1', 'income_1', 'educ_ba_1', 'white_1', 'black_1', 'native_1', 'asian_1'].sum().reset_index()
    com_sum_df = com_sum_df.rename(columns = {'population_1' : 'population_sum_1', 'income_1' : 'income_sum_1', 'educ_ba_1' : 'educ_ba_sum_1', 'white_1' : 'white_sum_1', 'black_1' : 'black_sum_1', 'native_1' : 'native_sum_1', 'asian_1' : 'asian_sum_1'})

    com_std_df = tract_df.groupby(['city','algorithm_type','g_type','S'])['population_1', 'income_1', 'educ_ba_1', 'white_1', 'black_1', 'native_1', 'asian_1'].std().reset_index()
    com_std_df = com_std_df.rename(columns = {'population_1' : 'population_std_1', 'income_1' : 'income_std_1', 'educ_ba_1' : 'educ_ba_std_1', 'white_1' : 'white_std_1', 'black_1' : 'black_std_1', 'native_1' : 'native_std_1', 'asian_1' : 'asian_std_1'})

    network_sum_df = tract_df.groupby(['city','algorithm_type','g_type'])['population_1', 'income_1', 'educ_ba_1', 'white_1', 'black_1', 'native_1', 'asian_1'].sum().reset_index()
    network_sum_df = network_sum_df.rename(columns = {'population_1' : 'population_sum_1', 'income_1' : 'income_sum_1', 'educ_ba_1' : 'educ_ba_sum_1', 'white_1' : 'white_sum_1', 'black_1' : 'black_sum_1', 'native_1' : 'native_sum_1', 'asian_1' : 'asian_sum_1'})

    network_std_df = tract_df.groupby(['city','algorithm_type','g_type'])['population_1', 'income_1', 'educ_ba_1', 'white_1', 'black_1', 'native_1', 'asian_1'].std().reset_index()
    network_std_df = network_std_df.rename(columns = {'population_1' : 'population_std_1', 'income_1' : 'income_std_1', 'educ_ba_1' : 'educ_ba_std_1', 'white_1' : 'white_std_1', 'black_1' : 'black_std_1', 'native_1' : 'native_std_1', 'asian_1' : 'asian_std_1'})

    community_df = pd.merge(community_df, com_sum_df, how = 'left', on = ['city','algorithm_type','g_type','S'])
    community_df = pd.merge(community_df, com_std_df, how = 'left', on = ['city','algorithm_type','g_type','S'])

    network_df = pd.merge(network_df, network_sum_df, how = 'left', on = ['city','algorithm_type','g_type'])
    network_df = pd.merge(network_df, network_std_df, how = 'left', on = ['city','algorithm_type','g_type'])
    return community_df, network_df

# 4.2.) INCOME PERCENTILE
#### scipy.stats.percentileofscore(array, score) - gives back the percentil of score
# top quality communities
def calc_income_percentile(tract_df, community_df):
    census_city_df = tract_df.groupby(['city','algorithm_type','g_type'])['income_1'].apply(list).reset_index()
    census_city_df = census_city_df.rename(columns = {'income_1' : 'income_1_l'})

    tract_df = pd.merge(tract_df, census_city_df[['city', 'income_1_l']], how = 'left', on = ['city'])
    community_df = pd.merge(community_df, census_city_df[['city', 'income_1_l']], how = 'left', on = ['city'])
    
    # KERDES EZ LENTEBB JO?
    tract_df['income_pct'] = tract_df.apply(lambda row: scipy.stats.percentileofscore(row['income_1_l'], row['income_1']),axis=1)
    community_df['income_pct'] = community_df.apply(lambda row: scipy.stats.percentileofscore(row['income_1_l'], (row['income_sum_1'] / row['tract_sum'])),axis=1) 
    return tract_df, community_df


# 4.3.) CREATE POOR TRACT DUMMIES - poor tracts are those with less than 50% of the median tract income in the given city

def create_poor_dummy(tract_df):
    """
    Create a poor dummy which takes the value 1 when the tract's income is lower than 50% of the median tract income in the network,
    and takes 0 otherwise. It is stored in tract_df.
    """

    # calculate median income in of each networks
    network_med_df = deepcopy(tract_df.groupby(['city','algorithm_type','g_type']))[
        'income_1'
        ].median().reset_index()
    network_med_df = network_med_df.rename(columns = {'income_1' : 'income_med_1'})  

    # relative poverty line - 50% of city's median income
    network_med_df['rel_poverty_line_1'] = network_med_df['income_med_1'] / 2
    tract_df = pd.merge(tract_df, network_med_df, how = 'left', on = ['city','algorithm_type','g_type'])

    # dummy for poor tracts
    tract_df['poor_1'] = (tract_df['income_1'] < tract_df['rel_poverty_line_1']).astype('int')
    return tract_df


# 4.4.) CALCULATE (PERCETAGE) DIFFERENCE OF TRACTS AND COMMUNITY AVERAGES FROM NETWORK AVERAGE

def diff_from_network_avg(tract_df, community_df, network_df):
    """

    Calculate the percentage difference (or raw difference in case of income) of tracts and communities from network average.

    It uses the dataframes tract_df and community_df
    and it adds average values to network_df and it adds new variables in tract_df and community_df such as:
    ['educ_ba_p_diff_1', 'white_p_diff_1', 'black_p_diff_1', 'native_p_diff_1', 'asian_p_diff_1', 'income_diff_1',
    and 'educ_ba_p_diff_2' ... etc. in tract_df

    and , 'white_p_diff_avg 1' etc. in the community_df

    Ex: 
    educ_ba_p_diff_1 MEANS: The percentage difference of population with a BA in the tract compared to the entire network.
    educ_ba_p_diff_avg_1 MEANS: The average percentage difference of population with a BA in the community compared to those in the entire network.

    """
    # calculate proportions - mean tract population is not the same among communities
    # for 2012
    tract_df['educ_ba_p_1'] = tract_df['educ_ba_1']/tract_df['population_1']
    tract_df['white_p_1'] = tract_df['white_1']/tract_df['population_1']
    tract_df['black_p_1'] = tract_df['black_1']/tract_df['population_1']
    tract_df['native_p_1'] = tract_df['native_1']/tract_df['population_1']
    tract_df['asian_p_1'] = tract_df['asian_1']/tract_df['population_1']
    # for 2017
    tract_df['educ_ba_p_2'] = tract_df['educ_ba_2']/tract_df['population_2']
    tract_df['white_p_2'] = tract_df['white_2']/tract_df['population_2']
    tract_df['black_p_2'] = tract_df['black_2']/tract_df['population_2']
    tract_df['native_p_2'] = tract_df['native_2']/tract_df['population_2']
    tract_df['asian_p_2'] = tract_df['asian_2']/tract_df['population_2']

    # calculate average percenateges in networks BY averaging its tracts (not weighted average - KERDES : ennyit kerekithetek, ugye?)
    network_avg_df = deepcopy(tract_df.groupby(['city','g_type'])['educ_ba_p_1','white_p_1','black_p_1','native_p_1','asian_p_1', 'income_1', 
                                                                  'educ_ba_p_2','white_p_2','black_p_2','native_p_2','asian_p_2', 'income_2'].mean().reset_index())

    network_avg_df = network_avg_df.rename(columns = {'educ_ba_p_1' : 'educ_ba_p_avg_1','white_p_1' : 'white_p_avg_1', 'black_p_1' : 'black_p_avg_1', 'native_p_1' : 'native_p_avg_1', 'asian_p_1' : 'asian_p_avg_1', 'income_1' : 'income_avg_1',
                                                      'educ_ba_p_2' : 'educ_ba_p_avg_2','white_p_2' : 'white_p_avg_2', 'black_p_2' : 'black_p_avg_2', 'native_p_2' : 'native_p_avg_2', 'asian_p_2' : 'asian_p_avg_2', 'income_2' : 'income_avg_2'})
    # merge network averages to tract_df and network_df
    tract_df = pd.merge(tract_df, network_avg_df, how = 'left', on = ['city', 'g_type'])
    network_df = pd.merge(network_df, network_avg_df, how = 'left', on = ['city', 'g_type'])

    # calculate ratio difference from network average
    # 2012
    tract_df['educ_ba_p_diff_1'] = tract_df['educ_ba_p_1'] - tract_df['educ_ba_p_avg_1']
    tract_df['white_p_diff_1'] = tract_df['white_p_1'] - tract_df['white_p_avg_1']
    tract_df['black_p_diff_1'] = tract_df['black_p_1'] - tract_df['black_p_avg_1']
    tract_df['native_p_diff_1'] = tract_df['native_p_1'] - tract_df['native_p_avg_1']
    tract_df['asian_p_diff_1'] = tract_df['asian_p_1'] - tract_df['asian_p_avg_1']
    tract_df['income_diff_1'] = tract_df['income_1'] - tract_df['income_avg_1']
    # 2017
    tract_df['educ_ba_p_diff_2'] = tract_df['educ_ba_p_2'] - tract_df['educ_ba_p_avg_2']
    tract_df['white_p_diff_2'] = tract_df['white_p_2'] - tract_df['white_p_avg_2']
    tract_df['black_p_diff_2'] = tract_df['black_p_2'] - tract_df['black_p_avg_2']
    tract_df['native_p_diff_2'] = tract_df['native_p_2'] - tract_df['native_p_avg_2']
    tract_df['asian_p_diff_2'] = tract_df['asian_p_2'] - tract_df['asian_p_avg_2']
    tract_df['income_diff_2'] = tract_df['income_2'] - tract_df['income_avg_2']


    community_avg_df = deepcopy(tract_df.groupby(['city','algorithm_type','g_type','S']))[
    'degree', 'population_1', 'educ_ba_1', 'income_1', 'white_1', 'black_1', 'native_1', 'asian_1', 
    'educ_ba_p_diff_1', 'white_p_diff_1', 'black_p_diff_1', 'native_p_diff_1', 'asian_p_diff_1', 'income_diff_1'    
    ].mean().reset_index()

    # kevesebb: 'educ_ba_p_diff_1', 'white_p_diff_1', 'black_p_diff_1', 'native_p_diff_1', 'asian_p_diff_1', 'income_diff_1'

    community_avg_df = community_avg_df.rename(columns = {
        'degree' : 'degree_avg', 'population_1' : 'population_avg_1', 'educ_ba_1' : 'educ_ba_avg_1', 'income_1' : 'income_avg_1',
        'white_1' : 'white_avg_1', 'black_1' : 'black_avg_1', 'native_1' : 'native_avg_1', 'asian_1' : 'asian_avg_1',
        'educ_ba_p_diff_1' : 'educ_ba_p_diff_avg_1', 'white_p_diff_1' : 'white_p_diff_avg_1', 'black_p_diff_1' : 'black_p_diff_avg_1',
        'native_p_diff_1' : 'native_p_diff_avg_1', 'asian_p_diff_1' : 'asian_p_diff_avg_1', 'income_diff_1' : 'income_diff_avg_1'        
        })
    # kevesebb
    community_df = pd.merge(community_df, community_avg_df, on = ['city','algorithm_type','g_type','S'])

    # TODO 0608 EZ ITT HIANYOSNAK TUNIK!!!!!!!!!!!!!!!!!!!!!!!!!!!4
    return tract_df, community_df



# df = pd.read_csv(....)

# def
# """
# """
#     df = ...


# ydf

# def

#     df.groupby()

    


# import functions as f
    
# f.functionname()

