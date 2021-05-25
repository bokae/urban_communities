        ## !!! Difference from previous notebooks: community labels (which are calculated from several runs, so they are consensus communities) are named 'S' INSTEAD OF 'S_cons' as it is shorter


# 0.) IMPORT

import pandas as pd
import geopandas as gpd
import json

import numpy as np
import scipy



# 1.) IMPORT DATA

# 1.1.) 3 networks

# 1.1.1.) aggregated to census tract level
mobility = pd.read_csv("../data/usageousers_city_mobility_CT_networks.rpt.gz") # basis of position and node importance calculations
follow_hh = pd.read_csv("../data/usageousers_city_follower_CT_HH_networks.rpt.gz")
follow_hh = follow_hh.rename(columns={"tract_home.1": "tract_home_1"})

# 1.1.2.) census tract name --> cbsacode
cbsacode = pd.read_csv("../data/cbsacode_shortname_tracts.csv",sep=";", index_col=0)
cbsacode['clean_name'] = cbsacode["short_name"].map(lambda s: s.split("/")[0].replace(' ','_').replace('.','').lower())


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
census = pd.read_csv("../data/censusdata_top50_2012.csv")
census_2 = pd.read_csv("../data/censusdata_top50_2017.csv")



# 2.) PREPARE DATA

# 2.1.) create tract geoids
def create_geoid(row):
    state = str(int(row["state"])).zfill(2)
    county = str(int(row["county"])).zfill(3)
    tract = str(int(row["tract"])).zfill(6)
    return "14000US" +state+county+tract

census['geoid'] = census.apply(create_geoid,axis=1)
census_2['geoid'] = census_2.apply(create_geoid,axis=1)

# 2.2.) list of city names
city_l = list(cbsacode['clean_name'].unique())


# 3.) AUXILIARY FUNCTION

###### START HERE, CHECK

# copied from Spatial_modularity_pooled notebook
def create_graph(city, g_type): ### ATIRTAM KICSIT, ELLENORIZNI
    """
    For a given city name, it generates a mobility and follower (home-home) graph.
    
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
        
    # TODO --> DONE
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


# 4.) NETWORK CALCULATIONS


# TODO ÁTSZÁMOZNI
# 4.0.---) Degree - TODO KIPROBALNI

def degree_degreecent_dict(city, g_type):
    # make the graph
    G = create_graph(city, g_type) # kell ez a sor??
    # calculate degree and degree centrality for each node
    degree_dict = dict(nx.degree(G, weight="cnt"))
    degreecent_values = np.array(list(degree_dict.values())) / np.array(list(degree_dict.values())).sum()
    degreecent_dict = dict(zip(degree.keys(), degreecent_values)) ## ezt így lehet??
    return degree_dict, degreecent_dict


# 4. --------) Density
# TODO



# 4.1.) Modularity calculations
## Ms/Mgn - deterrence function nem, mik ezek???? KERDES?
# calculations based on Newman-Girvan modularity (standard) and Expert modularity (spatially corrected: null model account for the fact that the closer two places to each other the more connection is expected between them)

# 4.1.1.) Calculation preparation --> ??? deterrence funtion matrix for each network for both methods ???


### START HERE
# copied from Spatial_modularity_pooled notebook
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
    
    #tic = time()
    
    #print("Beginning of modularity function...");   
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
    
    #toc = time()
    #print("Done.","%.2f" % (toc-tic))
    
    #tic = toc
    
    #print("Null modell...")
    
    # copmutation of the randomised correlations (preserving space), spatial
    # null-model
    nullmodelSpa = det[np.digitize(D,detbins,right=True)-1]
    
    #toc = time()
    #print("Done.","%.2f" % (toc-tic))
    
    #tic = toc
    
    #print("Modularity calc...")
    
    # the modularity matrix for the spatial null-model
    ModularitySpa=A-np.multiply(N*N.T, nullmodelSpa*A.sum())/(np.multiply(N*N.T,nullmodelSpa).sum())
    szamlalo = np.multiply(N*N.T, nullmodelSpa*A.sum())
    nevezo = np.multiply(N*N.T,nullmodelSpa).sum()
    
    # the modularity matrix for the GN null-model
    degree = degree = A.sum(axis=0) # JAVITVA np.squeeze(np.asarray(A.sum(axis=0))) # degree or strength of nodes || asarry for further usage
    nullmodelGN = degree.T*degree/degree.sum() # Newman-Girvan null-model
    ModularityGN = A - nullmodelGN
    
    #toc = time()
    #print("Done.","%.2f" % (toc-tic))
    
    return ModularitySpa, ModularityGN # KERDES UTOLAG: EZ AZT ADJA MEG, HOGY ADOTT KÉT CSÚCS KÖZÖTT MENNYIVEL TÖBB ÉL VAN A VÁRTNÁL??
# KERDES FOLYT.: EGY KLASZTEREZÉS MODULARITÁSÁT ÚGY KAPOM MEG EBBŐL, HOGY ÖSSZEADOM AZON ELEMEIT, AMIK MINDKÉT CSÚCSA UGYANAHHOZ A COMMUNITYHEZ TARTOZIK, MAJD EOLSZTOM 2M-MEL (2*m = 2-szer total weight of the network = 2*szum(A)/2 = szum(A)

### END HERE CHECK



## KERDES, HIBA

# INNEN

# HIBAS AZ ADATBAZIS --> vannak benne duplikátumok
algorithm_type = 'ms'
g_type = 'mob'
len(list(tract_df[(tract_df['algorithm_type'] == algorithm_type) & (tract_df['g_type'] == g_type)]['geoid'])) - len(set(tract_df[(tract_df['algorithm_type'] == algorithm_type) & (tract_df['g_type'] == g_type)]['geoid']))

# EZEKRE NEM FUT LE
# KERDES Ezekre miert nem mukodik?
city_l.remove('providence')
city_l.remove('san_jose')
city_l.remove('austin')
city_l.remove('san_francisco')
city_l.remove('washington')
city_l.remove('san_antonio')
city_l.remove('riverside')
city_l.remove('boston')
city_l.remove('baltimore')
city_l.remove('san_diego')
city_l.remove('los_angeles')

# EDDIG


# 4.1.2.) Modularity contribution of communities to the overall modularity of networks

### TODO: MODULARITY CONTRIBUTION NEM MÉRETFÜGGŐ?

### START

def community_modularity(city, g_type):
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
        S_notsorted = deepcopy(tract_df[(tract_df['city'] == city) & (tract_df['algorithm_type'] == algorithm_type) & (tract_df['g_type'] == g_type)].set_index('geoid')['S']) # KERDES Mikor kell deepcopy?
        S = [S_notsorted.loc[k] for k in G.nodes()] # KERDES: kell ez?
    
        for s in set(S):
            com_vector = (np.array(S) == s).astype('int')
            # clusters to matrix format
            # matrix for each cluster (in which 1s are for node pairs which are in the given community all others are 0s)
            # same community matrix (stored as array) - 0 if either of the nodes is not in community, 1 if both are in community
            same_com = numpy.outer(com_vector, com_vector)
      
            # add up how much more edges are present between community nodes then expected by nullmodel in the given community
            mod = (np.multiply(np.array(modularity), same_com).sum()) / (2 * A.sum())
            
            mod_df = mod_df.append({'city': city, 'algorithm_type' : algorithm_type, 'g_type' : g_type, 'S' : s, 'modularity_S' : mod}, ignore_index=True)
    
    return mod_df


### END CHECK







################## CREATE COMMUNITY_DF

# 6.) COMMUNITY LEVEL DATAFRAME

# creating the dataframe

# TODO PUT degree, degreecent and modularity into the tract_df
community_df = tract_df.groupby(['city', 'algorithm_type', 'g_type', 'S'])['degree', 'degreecent'].mean().reset_index()
community_df = community_df.rename(columns = {'degree' : 'degree_avg', 'degreecent' : 'degreecent_avg'})

# number of tracts or numer of users in it?? KERDES
# sum 'ones' or sum 'cnt'
community_cnt_df = tract_df.groupby(['city', 'algorithm_type', 'g_type', 'S'])['ones'].sum().reset_index()








# KERDES EZT HOGYAN??
com_mod_df = pd.DataFrame()
for city, g_type in graph_combs:
    print(city)
    print(g_type)
    df = community_modularity(city, g_type)
    com_mod_df = pd.concat([com_mod_df, df])
# TODO ADD TO COM_DF
community_df = pd.merge(community_df, com_mod_df, how = 'left', on = ['city', 'algorithm_type', 'g_type', 'S']) ## TODO megnézni, hogy hány sora marad, ha nem left, hanem inner, jó-e ez a merge

    
# 4.1.3.) Modularity of networks    

modularity_df = deepcopy(com_mod_df.groupby(['city', 'algorithm_type', 'g_type'])[['modularity_S']].sum().reset_index())
modularity_df = modularity_df.rename(columns = {'modularity_S' : 'modularity'})

# check this TODO
network_df = pd.merge(network_df, modularity_df, how = 'left', on = ['city', 'algorithm_type', 'g_type']) ### KERDES Mindig left merget kell csinálni? Ellenőrizni kell az üres értékeket?

# calculate modularity contribution = mod_community/mod_network
community_df = pd.merge(community_df, modularity_df, how = 'left', on = ['city', 'algorithm_type', 'g_type'])
commnity_df['mod_S_p'] = community_df['modularity_S'] / community_df['modularity']
    
    
    
# 4.1.4.) Community quality ?? (community's contribution to modularity of network / modularity of network)

# TODO IDERAKNI





# 5.) CENSUS CALCULATIONS - FOR CITY NOT FOR NETWORK!!!!!!

# NINCS PRÓBÁLVA ettől lejjebb
# censusok összerakni, _1 és _2 -t rárakni

#### scipy.stats.percentileofscore(array, score) - gives back the percentil of score


# top quality communities
census_city_df = census_df.groupby(['city'])['income_1'].apply(list).reset_index()
census_city_df = census_city_df.rename(columns = {'income_1' : 'income_1_l'})


tract_df = pd.merge(tract_df, census_city_df['income_1_l'], how = 'left', on = ['city'])
network_df = pd.merge(network_df, census_city_df['income_1_l'], how = 'left', on = ['city'])
 
# KERDES EZ LENTEBB JO?
tract_df['income_pct'] = tract_df.apply(lambda row: scipy.stats.percentileofscore(row['income_1_l'], row['income_1']))

network_df['income_pct'] = network_df.apply(lambda row: scipy.stats.percentileofscore(row['income_1_l'], row['income_1']))


'S'
# TODO EZELŐTT MÁG DEGREE, MODAULARITY ÉS SZÁMLÁLÓ OSZOP IS KELL





# elnevezes:
### TODO S_CONS HELYETT S, S HELYETT S_NONCONS
### educ_ba_1 legyen a neve
# SUM and STD
com_sum_df = tract_df.groupby(['city','algorithm_type','g_type','S'])['population_1', 'income_1', 'educ_ba_1', 'white_1', 'black_1', 'native_1', 'asian_1'].sum().reset_index()
com_sum_df = com_sum_df.rename(columns = {'population_1' : 'population_sum_1', 'income_1' : 'income_sum_1', 'educ_ba_1' : 'educ_ba_sum_1', 'white_1' : 'white_sum_1', 'black_1' : 'black_sum_1', 'native_1' : 'native_sum_1', 'asian_1' : 'asian_sum_1'})

com_std_df = tract_df.groupby(['city','algorithm_type','g_type','S'])['population_1', 'income_1', 'educ_ba_1', 'white_1', 'black_1', 'native_1', 'asian_1'].std().reset_index()
com_std_df = com_std_df.rename(columns = {'population_1' : 'population_std_1', 'income_1' : 'income_std_1', 'educ_ba_1' : 'educ_ba_std_1', 'white_1' : 'white_std_1', 'black_1' : 'black_std_1', 'native_1' : 'native_std_1', 'asian_1' : 'asian_std_1'})

city_sum_df = tract_df.groupby(['city','algorithm_type','g_type'])['population_1', 'income_1', 'educ_ba_1', 'white_1', 'black_1', 'native_1', 'asian_1'].sum().reset_index()
city_sum_df = city_sum_df.rename(columns = {'population_1' : 'population_sum_1', 'income_1' : 'income_sum_1', 'educ_ba_1' : 'educ_ba_sum_1', 'white_1' : 'white_sum_1', 'black_1' : 'black_sum_1', 'native_1' : 'native_sum_1', 'asian_1' : 'asian_sum_1'})

city_std_df = tract_df.groupby(['city','algorithm_type','g_type'])['population_1', 'income_1', 'educ_ba_1', 'white_1', 'black_1', 'native_1', 'asian_1'].std().reset_index()
city_std_df = city_std_df.rename(columns = {'population_1' : 'population_std_1', 'income_1' : 'income_std_1', 'educ_ba_1' : 'educ_ba_std_1', 'white_1' : 'white_std_1', 'black_1' : 'black_std_1', 'native_1' : 'native_std_1', 'asian_1' : 'asian_std_1'})


community_df = pd.merge(community_df, com_sum_df, how = 'left', on = ['city','algorithm_type','g_type','S'])
community_df = pd.merge(community_df, com_std_df, how = 'left', on = ['city','algorithm_type','g_type','S'])

network_df = pd.merge(network_df, city_sum_df, how = 'left', on = ['city','algorithm_type','g_type','S'])
network_df = pd.merge(network_df, city_std_df, how = 'left', on = ['city','algorithm_type','g_type','S'])










df = pd.read_csv(....)

def
"""
"""
    df = ...


ydf

def

    df.groupby()

    


import functions as f
    
f.functionname()

