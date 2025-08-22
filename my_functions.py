# -*- coding: utf-8 -*-
"""
Created on Thu May 30 10:38:25 2024

@author: chak803
"""

import math
import numpy as np
import pandas as pd
from tqdm import tqdm

def rao_quadratic_entropy(abundances, Traitdata):
    from scipy.spatial.distance import pdist, squareform

    pairwise_dissimilarities = pdist(Traitdata, metric='euclidean')

    abundances = np.array(abundances)
    pairwise_dissimilarities = np.array(pairwise_dissimilarities)
    
    # Normalize abundances
    p = abundances / abundances.sum()
    
    # Calculate pairwise dissimilarities
    D = squareform(pairwise_dissimilarities)
    
    # Calculate Rao's Quadratic Entropy
    rao_entropy = 0.5 * np.sum(p[:, None] * p[None, :] * D)
    
    return rao_entropy

def shannon_diversity_index(data):
    """
    Calculate the Shannon Diversity Index for a given dataset.
    
    Parameters:
    - data: List or array containing the relative abundances of different species or categories.
    
    Returns:
    - Shannon Diversity Index (float).
    """
    # Convert data to numpy array if it's not already
    data = np.array(data)
    
    # Calculate proportions of each category
    proportions = data / np.sum(data)
    
    # Filter out zero proportions to avoid log(0) errors
    nonzero_proportions = proportions[proportions > 0]
    
    # Calculate Shannon Diversity Index
    shannon_index = -np.sum(nonzero_proportions * np.log(nonzero_proportions))

    return shannon_index


#Aromaticity = DBE/(#Carbons -#double bonds from heteroatoms)
def AIcalc(C,H,N,O,S):#remove P
    """
    ### D. Calculate properties for each molecular formula <a name="calcprops"></a>
    
    #### Aromaticity Index
    
    The following section includes the code required to calculate various properties of each assigned formula based on its elemental composition.  
    
    First up is Aromaticity Index    
    
    $$AI = \\frac{1+C-O-S-0.5(N+P+H)}{C-O-N-S-P}$$
      
    Where each of C H O N S P represents the count of the respective elements. This equation assumes all O are C=O and do not
    contribute to aromaticity (rings or condensation).
    
    Reference - https://doi.org/10.1002/rcm.2386  
    Correction - https://doi.org/10.1002/rcm.7433 

    """
    top = 1+C-O-S-(0.5*(H+N)) 
    btm = C-O-S-N  #remove P
    AI = top / btm

    # if btm == 0:
    #     AI = 0
    # else:
    #     AI = top/btm
    # if AI < 0:
    #     AI = 0
    # if AI >1:
    #     AI = 1
    AI = AI.clip(0, 1)  # Clip values to range [0, 1]
    return AI


# Again, we exclude P here as we havent assigned it. 
def AImodcalc(C,H,N,O,S):#remove P
    O = O/2
    top = 1+C-O-S-(0.5*(H+N)) #corrected from 1/2 H only (0.5*H) per https://doi.org/10.1002/rcm.7433
    btm = C-O-S-N#remove P
    AI = top / btm

    # if btm == 0:
    #     AI = 0
    # else:
    #     AI = top/btm
    # if AI < 0:
    #     AI = 0
    # if AI >1:
    #     AI = 1
    AI = AI.clip(0, 1)  # Clip values to range [0, 1]
    return AI

def DBEcalc(C,H,N):#original DBE calculation, includes double bonds from heteroatoms but independent of O
    return 1+(0.5*(2*C-H+N))#remove P




# @jit(nopython=False)
def getThermoStoich_TEEM(chemForm, ui, pH_Input):

    """   
    
    This is the code for analysis of thermodynamic properties using python in a Jupyter Notebook template. 
    Also, a demo_input file has been provided with 7 compounds to run analysis for demo purpose or test purpose.
    As suggested by the demo_input content, you must have a C, H, N, O, P, S, Z titled columns in your to 
    be analyzed csv files to run the LambdaPy.
    

    Parameters:
    ----------
    - chemForm: vector of elemental composition
    - ui : float Enter a number (0,1.1,1.2, ...) from the provided list for a electron acceptor reaction. i.e. 0 is oxygen as a electron acceptor
    - pH_Input : pH value [0-14]

    Returns:
    ----------
    - df1: DataFrame with stiochiometry of all reactions, CUE, NUE, lambda values
    References:
    Adapted from https://github.com/hyunseobsong/lambda/blob/main/LambdaPy/LambdaPy.ipynb
    
    0) Oxygen (Default set to O2 as e acceptor)
    **0 (Default). O2 + 4H+ + 4e- ----> 2H2O**


    1) Nitrogen Compounds, Nitrates and Nitrites
    **1.1.  NO3- + 10H+ + 8e- ---> NH4+ + 3H2O %**

    **1.2.  NO3- + 2H+ + 2e- ---> NO2- + H2O**

    **1.3.  NO3- + 6H+ + 5e- ---> (1/2)N2 + 3H2O**

    **1.4   NO2- + 4H+ + 3e- ---> (1/2)N2 + 2H2O**

    **1.5.  NO2- + 8H+ + 6e- ---> NH4+ + 2H2O**

    **1.6.  N2 + 8H+ + 6e- ---> 2NH4+**


    2) Sulphur compounds, Sulphates and Sulphites
    **2.1.  SO4^2- + (9/2)H+ + 8e- ---> (1/2)H2S + (1/2)HS- + 4H2O**

    **2.2 SO4^2- + 2H+ + 2e- ---> SO3^2- + H2O**

    **2.3. SO4^2- + 5H+ + 4e- ---> (1/2)S2O3^2- + (5/2)H2O**

    **2.4. SO4^2- + 8H+ + 6e- ---> S + 4H2O**

    **2.5. SO4^2- + 9H+ + 8e- --> HS- + 4H2O**

    **2.6. SO3^2- + (15/2)H+ + 6e- ---> (1/2)H2S + (1/2)HS- + 3H2O**


    3) Iron compounds, ferrous and ferric
    **3.1. Fe(OH)3 + 3H+ + e- --> Fe2+ + 3H2O**

    **3.2. FeOOH + 3H+ + e- --> Fe2+ + 2H2O**

    **3.3. Fe3O4 + 8H+ + 2e- --> 3Fe2+ + 4H2O**

    **3.4.  Fe3+ + e- ---> Fe2+**


    4) Bicarbonate and Hydrogen ion
    **4.1. HCO3- + 9H+ + 8e- --> CH4 + 3H2O**

    **4.2. H+ + e- ---> (1/2)H2**


    5) Acetate
    **5  CH3COO- + 9H+ + 8e- --> 2CH4 + 2H2O**


    6) Manganese
    **6 MnO2 + 4H+ + 2e- --> Mn2+ + 2H2O**
    
    """
       
    # "OC"=C_a H_b N_c O_d P_e S_f^z
    # a,b,c,d,e,f,z = chemForm
    a = chemForm[0]
    b = chemForm[1]
    c = chemForm[2]
    d = chemForm[3]
    e = chemForm[4]
    f = chemForm[5]
    z = chemForm[6]


    stoich_colnames = ["donor", "h2o", "hco3", "nh4", "hpo4", "hs", "h", "e", "O2", "NO3", "NO2", "N2", "Fe3+", "Fe2+",
                       "H2", "SO4", "H2S", "SO3", "S", "S2O3", "Fe(OH)3", "FeOOH", "Fe3O4", "CH4", "CH3COO-", "MnO2", "Mn2+", "biom"]
    
    # Step 1a) stoichD: stoichiometries for an electron donor half reaction
    ySource = -1
    yH2o = -(3*a+4*e-d)
    yHco3 = a
    yNh4 = c
    yHpo4 = e
    yHs = f
    yH = 5*a+b-4*c-2*d+7*e-f
    yE = -z+4*a+b-3*c-2*d+5*e-2*f
    stoichD = np.zeros(28)
    stoichD[0:8] = [ySource, yH2o, yHco3, yNh4, yHpo4, yHs, yH, yE]
    stoichD[8:28] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    # stoichD_df = pd.DataFrame(stoichD, index=stoich_colnames)
    
    # Step 1b) stoichA: stoichiometries for an electron acceptor half reaction
    stoichA = np.zeros(28)
    stoichA[8] = -1  # oxygen
    stoichA[6] = -4  #  h+
    stoichA[7] = -4  #  e-
    stoichA[1] = 2  #  h2o
    # stoichA_df = pd.DataFrame(stoichA, index=stoich_colnames)

    if ui == 1.1:
        stoichA = np.zeros(28)
        stoichA[9] = -1
        stoichA[6] = -10
        stoichA[7] = -8
        stoichA[3] = 1
        stoichA[1] = 3

    elif ui == 1.5:
        stoichA = np.zeros(28)
        stoichA[10] = -1
        stoichA[6] = -8
        stoichA[7] = -6
        stoichA[3] = 1
        stoichA[1] = 2

    elif ui == 1.6:
        stoichA = np.zeros(28)
        stoichA[11] = -1
        stoichA[6] = -8
        stoichA[7] = -6
        stoichA[3] = 2
        stoichA[1] = 0

    elif ui == 3.4:
        stoichA = np.zeros(28)
        stoichA[12] = -1
        stoichA[6] = 0
        stoichA[7] = -1
        stoichA[13] = 1
        stoichA[1] = 0

    elif ui == 4.2:
        stoichA = np.zeros(28)
        stoichA[6] = -1
        stoichA[7] = -1
        stoichA[14] = 0.5

    elif ui == 1.2:
        stoichA = np.zeros(28)
        stoichA[9] = -1
        stoichA[6] = -2
        stoichA[7] = -2
        stoichA[10] = 1
        stoichA[1] = 1

    elif ui == 1.3:
        stoichA = np.zeros(28)
        stoichA[9] = -1
        stoichA[6] = -6
        stoichA[7] = -5
        stoichA[11] = 0.5
        stoichA[1] = 3

    elif ui == 1.4:
        stoichA = np.zeros(28)
        stoichA[10] = -1
        stoichA[6] = -4
        stoichA[7] = -3
        stoichA[11] = 0.5
        stoichA[1] = 2

    elif ui == 2.1:
        stoichA = np.zeros(28)
        stoichA[15] = -1
        stoichA[6] = -4.5
        stoichA[7] = -8
        stoichA[16] = 0.5
        stoichA[5] = 0.5
        stoichA[1] = 4

    elif ui == 2.6:
        stoichA = np.zeros(28)
        stoichA[17] = -1
        stoichA[6] = -7.5
        stoichA[7] = -6
        stoichA[16] = 0.5
        stoichA[5] = 0.5
        stoichA[1] = 3

    elif ui == 2.2:
        stoichA = np.zeros(28)
        stoichA[15] = -1
        stoichA[6] = -2
        stoichA[7] = -2
        stoichA[17] = 1
        stoichA[1] = 1

    elif ui == 2.4:
        stoichA = np.zeros(28)
        stoichA[15] = -1
        stoichA[6] = -8
        stoichA[7] = -6
        stoichA[18] = 1
        stoichA[1] = 4

    elif ui == 2.3:
        stoichA = np.zeros(28)
        stoichA[15] = -1
        stoichA[6] = -5
        stoichA[7] = -4
        stoichA[19] = 0.5
        stoichA[1] = 2.5

    elif ui == 0:
        stoichA = np.zeros(28)
        stoichA[8] = -1
        stoichA[6] = -4
        stoichA[7] = -4
        stoichA[1] = 2

    elif ui == 3.1:
        stoichA = np.zeros(28)
        stoichA[20] = -1
        stoichA[6] = -3
        stoichA[7] = -1
        stoichA[13] = 1
        stoichA[1] = 3

    elif ui == 3.2:
        stoichA = np.zeros(28)
        stoichA[21] = -1
        stoichA[6] = -3
        stoichA[7] = -1
        stoichA[13] = 1
        stoichA[1] = 2

    elif ui == 3.3:
        stoichA = np.zeros(28)
        stoichA[22] = -1
        stoichA[6] = -8
        stoichA[7] = -2
        stoichA[13] = 3
        stoichA[1] = 4

    elif ui == 2.5:
        stoichA = np.zeros(28)
        stoichA[15] = -1
        stoichA[6] = -9
        stoichA[7] = -8
        stoichA[5] = 1
        stoichA[1] = 4

    elif ui == 4.1:
        stoichA = np.zeros(28)
        stoichA[2] = -1
        stoichA[6] = -9
        stoichA[7] = -8
        stoichA[23] = 1
        stoichA[1] = 3

    elif ui == 5:
        stoichA = np.zeros(28)
        stoichA[24] = -1
        stoichA[6] = -9
        stoichA[7] = -8
        stoichA[23] = 2
        stoichA[1] = 2

    elif ui == 6:
        stoichA = np.zeros(28)
        stoichA[25] = -1
        stoichA[6] = -4
        stoichA[7] = -2
        stoichA[26] = 1
        stoichA[1] = 2

    # Step 1c) stoichCat: stoichiometries for catabolic reaction
    yEd = stoichD[7]
    yEa = stoichA[7]
    stoichCat = stoichD- ((yEd/yEa)*stoichA)

    # Step 2a) stoichAnStar: stoichiometries for anabolic reaction (N source = NH4+)
    chemFormBiom = [1, 1.8, 0.2, 0.5, 0, 0, 0] # C H_1.8 N_0.2 O_0.5
    aB = chemFormBiom[0]
    bB = chemFormBiom[1]
    cB = chemFormBiom[2]
    dB = chemFormBiom[3]
    eB = chemFormBiom[4]
    fB = chemFormBiom[5]
    zB = chemFormBiom[6]
    ySource = -1
    yH2o = -(3*aB+4*eB-dB)
    yHco3 = aB
    yNh4 = cB
    yHpo4 = eB
    yHs = fB
    yH = 5*aB+bB-4*cB-2*dB+7*eB-fB
    yE = -zB+4*aB+bB-3*cB-2*dB+5*eB-2*fB
    
    stoichAnStarB= np.zeros(28)
    stoichAnStarB[0:8] = [ySource, yH2o, yHco3, yNh4, yHpo4, yHs, yH, yE]
    stoichAnStarB[8:28] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0]  # add additional components: e-acceptor and biomass
    stoichAnStarB = - stoichAnStarB ##stoichi for biosynthesis (as opposite to biomass oxidation)
    stoichAnStarB[27] = stoichAnStarB[0] # change the location of biomass in the vector
    stoichAnStarB[0] = 0

    # Step 2b) "overall" anabolic reaction
    stoichAnStar = stoichAnStarB + (1/a)*stoichD

    yEana = stoichAnStar[7] #stoichi coeff for e of anabolism, it could be negative
    if yEana > 0:
        stoichAn = stoichAnStar-yEana/yEa*stoichA #require e acceptor
    elif yEana < 0:
        stoichAn = stoichAnStar-yEana/yEd*stoichD #required e donor
    else:
        stoichAn = stoichAnStar

     # Step 3: get lambda

    # - estimate delGd0 using LaRowe and Van Cappellen (2011)

    ne = -z+4*a+b-3*c-2*d+5*e-2*f  # number of electrons transferred in D
    nosc = -ne/a+4  # nominal oxidataion state of carbon
    delGcox0 = 60.3-28.5*nosc  # kJ/C-mol at 25 degC and 1 bar
    delGd0 = delGcox0*a*abs(stoichD[0])  # delG ED oxidation half reaction kJ/mol ED 

    # - estimate delGf0 (Gibbs energy of formation) for electron donor kJ/mol ED 
    delGf0_D_zero = 0 
    delGf0_zero = [delGf0_D_zero, -237.2, -586.9, -79.37, -1089.1, 12.05, 0, 0, 16.5, -111.3, -32.2, 18.19, -
                   4.6, -78.87, 0, -744.63, -33.4, -486.6, 0, 522.5, -690, -489.8, -1012.6, -34.06, -392, -465.14, -228, -67]
    
    delGcox0_zero = np.dot(delGf0_zero, stoichD) #energy of Rd half reaction without counting OC
    delGf0_D_est = (delGd0-delGcox0_zero)/stoichD[0] #estimated energy of OC (x*stoiD+delGcox0_zero=delGd0)
    delGf0 = delGf0_zero
    delGf0[0] = delGf0_D_est

    # delGf0_df = pd.DataFrame(delGf0, index=stoich_colnames)


    # - standard delG at pH=0
    delGcat0 = np.dot(delGf0, stoichCat)
    delGan0 = np.dot(delGf0, stoichAn)

    # - stadard delG at pH=7
    R = 0.008314
    T = 298.15
    iProton = 6 # index of proton in stoich_colnames
    proton_conc = 10**(-pH_Input)
    delGd = delGd0 + (R*T*stoichD[iProton])*(math.log(proton_conc)) ##e donor half reaction kJ/mol 
    delGcox = delGd / a ##e donor half reaction kJ/C-mol ED
    
    delGcat = delGcat0 + (R*T*stoichCat[iProton])*(math.log(proton_conc))  #kJ/C-mol ED
    delGan = delGan0 + (R*T*stoichAn[iProton])*(math.log(proton_conc))  #kJ/C-mol bio

    # The Thermodynamic Electron Equivalents Model (TEEM)
    eta = 0.43
    delGsyn = 200
    if math.isnan(delGan0) and math.isnan(delGan):
        lambda0 = math.nan
        lambda_ = math.nan
        stoichMet = [math.nan] * len(stoichCat)
        delGdis0 = math.nan
        delGdis = math.nan
    else:
        m = 1 if delGan < 0 else -1
        m0 = 1 if delGan0 < 0 else -1
        
        lambda0 = ((delGan0*(eta**m0))+delGsyn)/(-delGcat0*eta) # assume delGsyn0=delGsyn
        lambda_ = (delGan*eta**m+delGsyn)/(-delGcat*eta)
        if lambda_ > 0:
            stoichMet = lambda_*stoichCat + stoichAn
        else:
            stoichMet = stoichAn
        delGdis0 = np.dot(lambda0, -delGcat0) - delGan0
        delGdis = np.dot(lambda_, -delGcat) - delGan


    # Organizing all the calculated values in a dataframe
    CUE = stoichMet[-1] * 1 / (abs(stoichMet[0]) * a)
    NUE = stoichMet[-1] * 0.2 / (abs(stoichMet[0]) * c + abs(stoichMet[3]) * 1)
    TER = (NUE / CUE) * (1 / 0.2)

    stoich_colnames = ["donor", "h2o", "hco3", "nh4", "hpo4", "hs", "h", "e", "O2", "NO3", "NO2", "N2", "Fe3+", "Fe2+",
                       "H2", "SO4", "H2S", "SO3", "S", "S2O3", "Fe(OH)3", "FeOOH", "Fe3O4", "CH4", "CH3COO-", "MnO2", "Mn2+", "biom"]
    stoich_types = ["stoichD", "stoichA", "stoichCat", "stoichAn", "stoichMet"]
 
    names = ["NOSC","CUE", "NUE", "TER", "delGcox0", "delGd0", "delGcat0", "delGan0", "delGdis0", "lambda0",
                   "delGcox", "delGd", "delGcat", "delGan", "delGdis", "lambda"]
    names_extension = []
    for stoich_type in stoich_types:
        for colname in stoich_colnames:
            names_extension.append(f"{stoich_type}_{colname}")
    names.extend(names_extension)
       
    
    all_values = [nosc,CUE, NUE, TER, delGcox0, delGd0, delGcat0, delGan0,delGdis0, lambda0,
                  delGcox, delGd, delGcat, delGan, delGdis, lambda_]
    all_values.extend(stoichD)
    all_values.extend(stoichA)
    all_values.extend(stoichCat)
    all_values.extend(stoichAn)
    all_values.extend(stoichMet)
    df1 = pd.DataFrame(data=[all_values], columns=names)
    return df1

def getThermoStoich_Liu(chemForm, ui, pH_Input):

    """   
    
    This is the code for analysis of thermodynamic properties using python in a Jupyter Notebook template. 
    Also, a demo_input file has been provided with 7 compounds to run analysis for demo purpose or test purpose.
    As suggested by the demo_input content, you must have a C, H, N, O, P, S, Z titled columns in your to 
    be analyzed csv files to run the LambdaPy.
    

    Parameters:
    ----------
    - chemForm: vector of elemental composition
    - ui : float Enter a number (0,1.1,1.2, ...) from the provided list for a electron acceptor reaction. i.e. 0 is oxygen as a electron acceptor
    - pH_Input : pH value [0-14]

    Returns:
    ----------
    - df1: DataFrame with stiochiometry of all reactions, CUE, NUE, lambda values
    References:
    Adapted from https://github.com/hyunseobsong/lambda/blob/main/LambdaPy/LambdaPy.ipynb
    
    0) Oxygen (Default set to O2 as e acceptor)
    **0 (Default). O2 + 4H+ + 4e- ----> 2H2O**


    1) Nitrogen Compounds, Nitrates and Nitrites
    **1.1.  NO3- + 10H+ + 8e- ---> NH4+ + 3H2O %**

    **1.2.  NO3- + 2H+ + 2e- ---> NO2- + H2O**

    **1.3.  NO3- + 6H+ + 5e- ---> (1/2)N2 + 3H2O**

    **1.4   NO2- + 4H+ + 3e- ---> (1/2)N2 + 2H2O**

    **1.5.  NO2- + 8H+ + 6e- ---> NH4+ + 2H2O**

    **1.6.  N2 + 8H+ + 6e- ---> 2NH4+**


    2) Sulphur compounds, Sulphates and Sulphites
    **2.1.  SO4^2- + (9/2)H+ + 8e- ---> (1/2)H2S + (1/2)HS- + 4H2O**

    **2.2 SO4^2- + 2H+ + 2e- ---> SO3^2- + H2O**

    **2.3. SO4^2- + 5H+ + 4e- ---> (1/2)S2O3^2- + (5/2)H2O**

    **2.4. SO4^2- + 8H+ + 6e- ---> S + 4H2O**

    **2.5. SO4^2- + 9H+ + 8e- --> HS- + 4H2O**

    **2.6. SO3^2- + (15/2)H+ + 6e- ---> (1/2)H2S + (1/2)HS- + 3H2O**


    3) Iron compounds, ferrous and ferric
    **3.1. Fe(OH)3 + 3H+ + e- --> Fe2+ + 3H2O**

    **3.2. FeOOH + 3H+ + e- --> Fe2+ + 2H2O**

    **3.3. Fe3O4 + 8H+ + 2e- --> 3Fe2+ + 4H2O**

    **3.4.  Fe3+ + e- ---> Fe2+**


    4) Bicarbonate and Hydrogen ion
    **4.1. HCO3- + 9H+ + 8e- --> CH4 + 3H2O**

    **4.2. H+ + e- ---> (1/2)H2**


    5) Acetate
    **5  CH3COO- + 9H+ + 8e- --> 2CH4 + 2H2O**


    6) Manganese
    **6 MnO2 + 4H+ + 2e- --> Mn2+ + 2H2O**
    
    """
       
    # "OC"=C_a H_b N_c O_d P_e S_f^z
    a,b,c,d,e,f,z = chemForm
    # a = chemForm[0]
    # b = chemForm[1]
    # c = chemForm[2]
    # d = chemForm[3]
    # e = chemForm[4]
    # f = chemForm[5]
    # z = chemForm[6]

    stoich_colnames = ["donor", "h2o", "hco3", "nh4", "hpo4", "hs", "h", "e", "O2", "NO3", "NO2", "N2", "Fe3+", "Fe2+",
                       "H2", "SO4", "H2S", "SO3", "S", "S2O3", "Fe(OH)3", "FeOOH", "Fe3O4", "CH4", "CH3COO-", "MnO2", "Mn2+", "biom"]
    
    # Step 1a) stoichD: stoichiometries for an electron donor half reaction
    ySource = -1
    yH2o = -(3*a+4*e-d)
    yHco3 = a
    yNh4 = c
    yHpo4 = e
    yHs = f
    yH = 5*a+b-4*c-2*d+7*e-f
    yE = -z+4*a+b-3*c-2*d+5*e-2*f
    stoichD = np.zeros(28)
    stoichD[0:8] = [ySource, yH2o, yHco3, yNh4, yHpo4, yHs, yH, yE]
    stoichD[8:28] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    # stoichD_df = pd.DataFrame(stoichD, index=stoich_colnames)
    
    # Step 1b) stoichA: stoichiometries for an electron acceptor half reaction
    stoichA = np.zeros(28)
    stoichA[8] = -1  # oxygen
    stoichA[6] = -4  #  h+
    stoichA[7] = -4  #  e-
    stoichA[1] = 2  #  h2o
    # stoichA_df = pd.DataFrame(stoichA, index=stoich_colnames)

    if ui == 1.1:
        stoichA = np.zeros(28)
        stoichA[9] = -1
        stoichA[6] = -10
        stoichA[7] = -8
        stoichA[3] = 1
        stoichA[1] = 3

    elif ui == 1.5:
        stoichA = np.zeros(28)
        stoichA[10] = -1
        stoichA[6] = -8
        stoichA[7] = -6
        stoichA[3] = 1
        stoichA[1] = 2

    elif ui == 1.6:
        stoichA = np.zeros(28)
        stoichA[11] = -1
        stoichA[6] = -8
        stoichA[7] = -6
        stoichA[3] = 2
        stoichA[1] = 0

    elif ui == 3.4:
        stoichA = np.zeros(28)
        stoichA[12] = -1
        stoichA[6] = 0
        stoichA[7] = -1
        stoichA[13] = 1
        stoichA[1] = 0

    elif ui == 4.2:
        stoichA = np.zeros(28)
        stoichA[6] = -1
        stoichA[7] = -1
        stoichA[14] = 0.5

    elif ui == 1.2:
        stoichA = np.zeros(28)
        stoichA[9] = -1
        stoichA[6] = -2
        stoichA[7] = -2
        stoichA[10] = 1
        stoichA[1] = 1

    elif ui == 1.3:
        stoichA = np.zeros(28)
        stoichA[9] = -1
        stoichA[6] = -6
        stoichA[7] = -5
        stoichA[11] = 0.5
        stoichA[1] = 3

    elif ui == 1.4:
        stoichA = np.zeros(28)
        stoichA[10] = -1
        stoichA[6] = -4
        stoichA[7] = -3
        stoichA[11] = 0.5
        stoichA[1] = 2

    elif ui == 2.1:
        stoichA = np.zeros(28)
        stoichA[15] = -1
        stoichA[6] = -4.5
        stoichA[7] = -8
        stoichA[16] = 0.5
        stoichA[5] = 0.5
        stoichA[1] = 4

    elif ui == 2.6:
        stoichA = np.zeros(28)
        stoichA[17] = -1
        stoichA[6] = -7.5
        stoichA[7] = -6
        stoichA[16] = 0.5
        stoichA[5] = 0.5
        stoichA[1] = 3

    elif ui == 2.2:
        stoichA = np.zeros(28)
        stoichA[15] = -1
        stoichA[6] = -2
        stoichA[7] = -2
        stoichA[17] = 1
        stoichA[1] = 1

    elif ui == 2.4:
        stoichA = np.zeros(28)
        stoichA[15] = -1
        stoichA[6] = -8
        stoichA[7] = -6
        stoichA[18] = 1
        stoichA[1] = 4

    elif ui == 2.3:
        stoichA = np.zeros(28)
        stoichA[15] = -1
        stoichA[6] = -5
        stoichA[7] = -4
        stoichA[19] = 0.5
        stoichA[1] = 2.5

    elif ui == 0:
        stoichA = np.zeros(28)
        stoichA[8] = -1
        stoichA[6] = -4
        stoichA[7] = -4
        stoichA[1] = 2

    elif ui == 3.1:
        stoichA = np.zeros(28)
        stoichA[20] = -1
        stoichA[6] = -3
        stoichA[7] = -1
        stoichA[13] = 1
        stoichA[1] = 3

    elif ui == 3.2:
        stoichA = np.zeros(28)
        stoichA[21] = -1
        stoichA[6] = -3
        stoichA[7] = -1
        stoichA[13] = 1
        stoichA[1] = 2

    elif ui == 3.3:
        stoichA = np.zeros(28)
        stoichA[22] = -1
        stoichA[6] = -8
        stoichA[7] = -2
        stoichA[13] = 3
        stoichA[1] = 4

    elif ui == 2.5:
        stoichA = np.zeros(28)
        stoichA[15] = -1
        stoichA[6] = -9
        stoichA[7] = -8
        stoichA[5] = 1
        stoichA[1] = 4

    elif ui == 4.1:
        stoichA = np.zeros(28)
        stoichA[2] = -1
        stoichA[6] = -9
        stoichA[7] = -8
        stoichA[23] = 1
        stoichA[1] = 3

    elif ui == 5:
        stoichA = np.zeros(28)
        stoichA[24] = -1
        stoichA[6] = -9
        stoichA[7] = -8
        stoichA[23] = 2
        stoichA[1] = 2

    elif ui == 6:
        stoichA = np.zeros(28)
        stoichA[25] = -1
        stoichA[6] = -4
        stoichA[7] = -2
        stoichA[26] = 1
        stoichA[1] = 2

    # Step 1c) stoichCat: stoichiometries for catabolic reaction
    yEd = stoichD[7]
    yEa = stoichA[7]
    stoichCat = stoichD- ((yEd/yEa)*stoichA)
    stoichCat = stoichD - (yEd/yEa)*stoichA

    # Step 2a) stoichAnStar: stoichiometries for anabolic reaction (N source = NH4+)
    chemFormBiom = [1, 1.8, 0.2, 0.5, 0, 0, 0] # C H_1.8 N_0.2 O_0.5
    aB = chemFormBiom[0]
    bB = chemFormBiom[1]
    cB = chemFormBiom[2]
    dB = chemFormBiom[3]
    eB = chemFormBiom[4]
    fB = chemFormBiom[5]
    zB = chemFormBiom[6]
    ySource = -1
    yH2o = -(3*aB+4*eB-dB)
    yHco3 = aB
    yNh4 = cB
    yHpo4 = eB
    yHs = fB
    yH = 5*aB+bB-4*cB-2*dB+7*eB-fB
    yE = -zB+4*aB+bB-3*cB-2*dB+5*eB-2*fB
    
    stoichAnStarB= np.zeros(28)
    stoichAnStarB[0:8] = [ySource, yH2o, yHco3, yNh4, yHpo4, yHs, yH, yE]
    stoichAnStarB[8:28] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0]  # add additional components: e-acceptor and biomass
    stoichAnStarB = - stoichAnStarB ##stoichi for biosynthesis (as opposite to biomass oxidation)
    stoichAnStarB[27] = stoichAnStarB[0] # change the location of biomass in the vector
    stoichAnStarB[0] = 0

    # Step 2b) "overall" anabolic reaction
    stoichAnStar = stoichAnStarB + (1/a)*stoichD

    yEana = stoichAnStar[7] #stoichi coeff for e of anabolism, it could be negative
    if yEana > 0:
        stoichAn = stoichAnStar-yEana/yEa*stoichA #require e acceptor
    elif yEana < 0:
        stoichAn = stoichAnStar-yEana/yEd*stoichD #required e donor
    else:
        stoichAn = stoichAnStar


     # Step 3: get lambda

    # - estimate delGd0 using LaRowe and Van Cappellen (2011)

    ne = -z+4*a+b-3*c-2*d+5*e-2*f  # number of electrons transferred in D
    nosc = -ne/a+4  # nominal oxidataion state of carbon
    delGcox0 = 60.3-28.5*nosc  # kJ/C-mol at 25 degC and 1 bar
    delGd0 = delGcox0*a*abs(stoichD[0])  # delG ED oxidation half reaction kJ/mol ED 

    # - estimate delGf0 (Gibbs energy of formation) for electron donor kJ/mol ED 
    delGf0_D_zero = 0 
    delGf0_zero = [delGf0_D_zero, -237.2, -586.9, -79.37, -1089.1, 12.05, 0, 0, 16.5, -111.3, -32.2, 18.19, -
                   4.6, -78.87, 0, -744.63, -33.4, -486.6, 0, 522.5, -690, -489.8, -1012.6, -34.06, -392, -465.14, -228, -67]
    
    delGcox0_zero = np.dot(delGf0_zero, stoichD) #energy of Rd half reaction without counting OC
    delGf0_D_est = (delGd0-delGcox0_zero)/stoichD[0] #estimated energy of OC (x*stoiD+delGcox0_zero=delGd0)
    delGf0 = delGf0_zero
    delGf0[0] = delGf0_D_est

    # delGf0_df = pd.DataFrame(delGf0, index=stoich_colnames)


    # - standard delG at pH=0
    delGcat0 = np.dot(delGf0, stoichCat)
    delGan0 = np.dot(delGf0, stoichAn)

    # - stadard delG at pH=7
    R = 0.008314
    T = 298.15
    iProton = 6 # index of proton in stoich_colnames
    proton_conc = 10**(-pH_Input)
    delGd = delGd0 + (R*T*stoichD[iProton])*(math.log(proton_conc)) ##e donor half reaction kJ/mol 
    delGcox = delGd / a ##e donor half reaction kJ/C-mol ED
    
    delGcat = delGcat0 + (R*T*stoichCat[iProton])*(math.log(proton_conc))  #kJ/C-mol ED
    delGan = delGan0 + (R*T*stoichAn[iProton])*(math.log(proton_conc))  #kJ/C-mol bio

   
    # delGdis0 calculation based on Liu et al. (2007) doi: 10.1016/j.tca.2007. 01.016
    DR = 4-nosc
    if DR < 4.67:
        delGdis = (666.7 / DR + 243.1)  # kJ/Cmol biomass delGdis = -delGmet
    else:
        delGdis = -(-157 * DR + 339)  # kJ/Cmol biomass
    lambda0 = (delGdis + delGan0)/(-delGcat0) # assume delGdis0=delGdis
    lambda_ = (delGdis + delGan)/(-delGcat)
    if lambda_ > 0:
        stoichMet = lambda_*stoichCat + stoichAn
    else:
        stoichMet = stoichAn
    delGdis0 = np.dot(lambda0, -delGcat0) - delGan0
    delGdis = np.dot(lambda_, -delGcat) - delGan

    # Organizing all the calculated values in a dataframe
    CUE = stoichMet[-1] * 1 / (abs(stoichMet[0]) * a)
    NUE = stoichMet[-1] * 0.2 / (abs(stoichMet[0]) * c + abs(stoichMet[3]) * 1)
    TER = (NUE / CUE) * (1 / 0.2)
    

    stoich_colnames = ["donor", "h2o", "hco3", "nh4", "hpo4", "hs", "h", "e", "O2", "NO3", "NO2", "N2", "Fe3+", "Fe2+",
                       "H2", "SO4", "H2S", "SO3", "S", "S2O3", "Fe(OH)3", "FeOOH", "Fe3O4", "CH4", "CH3COO-", "MnO2", "Mn2+", "biom"]
    stoich_types = ["stoichD", "stoichA", "stoichCat", "stoichAn", "stoichMet"]
 
    names = ["NOSC","CUE", "NUE", "TER", "delGcox0", "delGd0", "delGcat0", "delGan0", "delGdis0", "lambda0",
                   "delGcox", "delGd", "delGcat", "delGan", "delGdis", "lambda"]
    names_extension = []
    for stoich_type in stoich_types:
        for colname in stoich_colnames:
            names_extension.append(f"{stoich_type}_{colname}")
    names.extend(names_extension)
       
    
    all_values = [nosc,CUE, NUE, TER, delGcox0, delGd0, delGcat0, delGan0,delGdis0, lambda0,
                  delGcox, delGd, delGcat, delGan, delGdis, lambda_]
    all_values.extend(stoichD)
    all_values.extend(stoichA)
    all_values.extend(stoichCat)
    all_values.extend(stoichAn)
    all_values.extend(stoichMet)
    df1 = pd.DataFrame(data=[all_values], columns=names)
    return df1



def get_compositions(df):
    CHEMICAL_ELEMENTS = ["C", "H", "N", "O", "P", "S"]
    chemical_compositions = None
    # formulas = None
    if "C" in df.columns:
        tdf = df[df["C"] > 0]
        try:
            if "C13" in df.columns:
                tdf = tdf[tdf["C13"] == 0]
        except:
            print("C13 not present in chemical formula")
        
        chemical_compositions = np.array(tdf[CHEMICAL_ELEMENTS])
        # formulas = tdf["MolForm"].tolist()
        pH_Input = tdf['pH']       

    else:
        raise ValueError("Either columns for compositions (e.g., C, H, N, ...) column is required.")
    if "Z" in df.columns:
        chemical_compositions = np.column_stack((chemical_compositions, df["Z"]))
    else:
        chemical_compositions = np.column_stack((chemical_compositions, np.zeros(len(df))))
    return {
        "chemical_compositions": chemical_compositions,
        # "formulas": formulas,
        "pH_Input":pH_Input
    }



def get_lambda(gt, ui,lambda_method):
    compositions=gt["chemical_compositions"]
    pH_Input = gt['pH_Input']
    result_list = []
    if lambda_method == "TEEM":
        lambda_fun = getThermoStoich_TEEM
    elif lambda_method == "Liu et al. (2007)":
        lambda_fun = getThermoStoich_Liu
    else:
        raise ValueError("lambda_method not specified")
    
    for i, pH in tqdm(zip(compositions,pH_Input)):
        b = lambda_fun(i, ui, pH)
        result_list.append(b)
    # Concatenate all DataFrames at once
    b1 = pd.concat(result_list, ignore_index=True)
    return b1


def assign_class(df, boundary_set):
    """
    Assigns a compound class to each peak/mass, where possible, based Oxygen:Carbon ratio and Hydrogen:Carbon ratio and a chosen boundary set.

    Parameters
    ----------
    df : dataframe with H/C and O/C columns
    boundary_set : TYPE
        DESCRIPTION.

    Returns
    -------
    compound_classes : TYPE
        DESCRIPTION.
    @reference bs1: Kim, S., Kramer, R. W., & Hatcher, P. G. (2003). Graphical method for analysis of ultrahigh-resolution broadband mass spectra of natural organic matter, the van Krevelen diagram. Analytical Chemistry, 75(20), 5336-5344.
    bs2: Bailey, V. L., Smith, A. P., Tfaily, M., Fansler, S. J., & Bond-Lamberty, B. (2017). Differences in soluble organic carbon chemistry in pore waters sampled from different pore size domains. Soil Biology and Biochemistry, 107, 133-143.
    bs3: Rivas-Ubach, A., Liu, Y., Bianchi, T. S., Toliċ, N., Jansson, C., & Paša-Tolić, L. (2018). Moving beyond the van Krevelen diagram: A new stoichiometric approach for compound classification in organisms. Analytical chemistry. DOI: 10.1021/acs.analchem.8b00529 
    
    Adapted from https://github.com/EMSL-Computing/ftmsRanalysis/blob/master/R/assign_class.R
    https://github.com/EMSL-Computing/ftmsRanalysis/blob/master/R/getVanKrevelenCategories.R
    """
    
    if boundary_set=="bs1":        
        bs1 = {
            "HC.low": [1.55, 0.7, 1.45, 0.81, 1.48, 1.34, 0.7, 0.3, 0],
            "HC.high": [2.25, 1.5, 2, 1.45, 2.15, 1.8, 1.3, 0.81, float('inf')],
            "OC.low": [0, 0.05, 0.3, 0.28, 0.68, 0.54, 0.65, 0.12, 0],
            "OC.high": [0.3, 0.15, 0.55, 0.65, 1, 0.71, 1.05, 0.7, float('inf')],
            "Category": ["Lipid-like", "Unsat Hydrocarbon-like", "Protein-like", "Lignin-like",
                          "Carbohydrate-like", "Amino Sugar-like", "Tannin-like", "Cond Hydrocarbon-like", "Other"]
        }
        bound_match = pd.DataFrame(bs1)

    elif boundary_set=="bs2":
        bs2= {
            "HC.low": [ 1.5,  0.8,  1.5,  0.8,  1.5,  1.5,  0.8,  0.2,  0],
            "HC.high": [2.5, 2.5, 2.3, 1.5, 2.5, 2.2, 1.5, 0.8, float('inf')],
            "OC.low": [0,  0, 0.3, 0.125, 0.7, 0.55, 0.65,  0,  0],
            "OC.high": [0.3, 0.125, 0.55, 0.65, 1.5, 0.7, 1.1, 0.95, float('inf')],
            "Category": ["Lipid-like","Unsat Hydrocarbon-like", "Protein-like","Lignin-like",
                         "Carbohydrate-like","Amino Sugar-like","Tannin-like","Cond Hydrocarbon-like","Other"]
            }
        bound_match = pd.DataFrame(bs2)
        
    bound_match.index = bound_match["Category"]
    bound_match = bound_match.drop(columns=["Category"])

    compound_classes = []
    for index, row in tqdm(df.iterrows()):
        hc_ratio = row['H/C']
        oc_ratio = row['O/C']
        found = False
        for category, bounds in bound_match.iterrows():
            if bounds['HC.low'] <= hc_ratio <= bounds['HC.high'] and bounds['OC.low'] <= oc_ratio <= bounds['OC.high']:
                compound_classes.append(category)
                found = True
                break
        if not found:
            compound_classes.append('Other')
    return compound_classes
