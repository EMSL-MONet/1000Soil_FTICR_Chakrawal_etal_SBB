# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 17:06:13 2024

@author: chak803
"""
# %%
# import libraries
import matplotlib.pyplot as plt
import matplotlib as mlp
import seaborn as sns
import numpy as np 
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from tqdm import tqdm
import plotly.express as px
import shap
import statsmodels.api as sm
from my_functions import (
    assign_class,
    shannon_diversity_index,
    get_compositions,
    get_lambda,
    rao_quadratic_entropy,
    AImodcalc,
    DBEcalc,
)

# Set up color map and seaborn style
color_map = mlp.colormaps.get_cmap('tab10')  # Qualitatively different colors
sns.set(style="whitegrid", font_scale=1.1)
sns.set_palette("colorblind")

#%% Global options

run_lambda_model=False # True to run the lambda model on ICR data, False to use pre-computed data

# %% load 1000 soil biogeochemical data and clean up

metadata = pd.read_excel("1000Soil_data/1000Soils_Metadata_Site_Mastersheet_v1.xlsx")
metadata['Site'] = metadata['Site_Code'].str.replace(' ', '')

df_BG = pd.read_excel("1000Soil_data/1000S_Dataset_Biogeochem_Biomass_Tomography_WEOM_2023_06_12.xlsx")
df_BG.insert(1, 'Site', df_BG['Sample_ID'].str.split('_').str[1])
df_BG['Sample'] = df_BG['Site']+"_"+df_BG['Location']

df_BG = df_BG.merge(metadata, how='outer', on='Site')

wsu_rows = df_BG['Site'].str.contains("WSU", case=False, na=False)
df_BG.loc[wsu_rows, ['Lat', 'Long']] = [46.2523, -119.737]
df_BG.loc[df_BG['Lat'].isna(), 'Site']

OAES_rows = df_BG['Site'].isin(['OAES'])
df_BG.loc[OAES_rows, ['Lat', 'Long']] = [35.410599, -99.058779]

TEAK_rows = df_BG['Site'].isin(['TEAK'])
df_BG.loc[TEAK_rows, ['Lat', 'Long']] = [37.00583, -119.00602]

# Define Biome Dictionaries
biome_dict = {  # 'Desert_XericShrubland'
    'ANZA': 'Desert shrubland',
    'FTA3': 'Desert shrubland',
    'FTA5': 'Desert shrubland',
    'JORN': 'Desert shrubland',
    'MOAB': 'Desert shrubland',
    'OCTB': 'Desert shrubland',
    'OCTU': 'Desert shrubland',
    'ONAQ': 'Desert shrubland',
    'PRS2': 'Desert shrubland',
    'SRER': 'Desert shrubland',
    'SRR1': 'Desert shrubland',
    'UT32': 'Desert shrubland',
    'WSU1': 'Desert shrubland',
    'WSU2': 'Desert shrubland',
    'WSU3': 'Desert shrubland',
    'OAES': 'Desert shrubland',

    # PUUM #missing from FTICRMS

    # 'Mediterraneanforests_Woodlands_Scrub'
    'SJER': 'Mediterranean woodlands',

    # : 'Temperate_Broadleaf_&_Mixed forests'
    'BLAN': 'Temperate forests',
    'CFS2': 'Temperate forests',
    'DELA': 'Temperate forests',
    'GRSM': 'Temperate forests',
    'LENO': 'Temperate forests',
    'MLBS': 'Temperate forests',
    'NWBA': 'Temperate forests',
    'NWBB': 'Temperate forests',
    'NWBC': 'Temperate forests',
    'ORNL': 'Temperate forests',
    'PPRH': 'Temperate forests',
    'PRS1': 'Temperate forests',
    'SCBI': 'Temperate forests',
    'SERC': 'Temperate forests',
    'TALL': 'Temperate forests',
    'WLLO': 'Temperate forests',
    'WLUP': 'Temperate forests',
    # HARV #missing from FTICRMS
    # STEI #missing from FTICRMS
    # TREE #missing from FTICRMS
    # Temperate_Conifer_forests
    'CFS1': 'Temperate coniferous forests',
    'DSNY': 'Temperate coniferous forests',
    'JERC': 'Temperate coniferous forests',
    'OKPF': 'Temperate coniferous forests',
    'OSBS': 'Temperate coniferous forests',
    'PETF': 'Temperate coniferous forests',
    'PHTU': 'Temperate coniferous forests',
    'RMNP': 'Temperate coniferous forests',
    'SOAP': 'Temperate coniferous forests',
    'UT12': 'Temperate coniferous forests',
    'UT19': 'Temperate coniferous forests',
    'UT23': 'Temperate coniferous forests',
    'UT24': 'Temperate coniferous forests',
    'WY01': 'Temperate coniferous forests',
    'WY03': 'Temperate coniferous forests',
    'WY09': 'Temperate coniferous forests',
    'WY10': 'Temperate coniferous forests',
    'WY15': 'Temperate coniferous forests',
    'TEAK': 'Temperate coniferous forests',
    # NIWO #missing from FTICRMS


    # Temperate_Grasslands_Savannas_Shrublands
    'CLBJ': 'Temperate grasslands',
    'CPER': 'Temperate grasslands',
    'DCFS': 'Temperate grasslands',
    'ISCC': 'Temperate grasslands',
    'ISNC': 'Temperate grasslands',
    'KONA': 'Temperate grasslands',
    'KONZ': 'Temperate grasslands',
    'NOGP': 'Temperate grasslands',
    'UKFS': 'Temperate grasslands',
    'STER': 'Temperate grasslands',
    # WOOD #missing from ICRMS
}

df_BG = df_BG.drop(columns=[
                      'Respiration_96_h_ug_CO2-C_per_g_soil_per_day',  # lots of nan
                      # 'B-glucosidase_ug_pNP_per_g_per_hour',  # lots of nan and it is correlated with MBC so include MBC
                      'B-glucosidase_activity_nmol_B-gluc_per_g_soil_per_hour',
                      'ROI.volume.xyz',
                      'Voxel.size',
                      'pore.size.distribution_0-0.1mm',
                      'pore.size.distribution_0.1-0.2mm',
                      'pore.size.distribution_0.2-0.3mm',
                      'pore.size.distribution_0.3-0.4mm',
                      'pore.size.distribution_0.4-0.5mm',
                      'pore.size.distribution_0.5-0.75mm',
                      'pore.size.distribution_0.75-1mm',
                      'pore.size.distribution_1.0-5.0mm',
                      'pore.size.distribution_5.0-10.0mm',
                      'pore.size.distribution_10.0-25.0mm',
                      'pore.size.distribution_25.0-50.0mm',
                      'pore.size.distribution_>50.0mm',
                      'pore.size.min',
                      'pore.size.max',
                      'pore.size.mean',
                      'pore.size.median',
                      'pore.size.variance',
                      'pore.volume.mean',
                      'pore.volume.fraction',
                      'wet.bulk.density',
                      'pnm.abs.permeability_x',
                      'pnm.total.flow.rate_x',
                      'pnm.tortuosity_x',
                      'pnm.abs.permeability_y',
                      'pnm.total.flow.rate_y',
                      'pnm.tortuosity_y',
                      'pnm.abs.permeability_z',
                      'pnm.total.flow.rate_z',
                      'pnm.tortuosity_z',
                      'Core_collector',
                      'DateTime_Collected (24 Hr format)',
                      'Vegetation',
                      'Weather',
                      'Site_Name_Long',
                      'connected.pores',
                      'VolWaterContent.m3/m3',
                      'TKN_pct'  # same as Total_Nitrogen_pct
                      # 'salt_C_mean', 'salt_N_mean' # remove this becuae hihly correlated with MBC and MBN, and i want to keep MBC
                      ])

df_BG['BiomeType'] = df_BG['Site'].map(biome_dict)


# df_BG=df_BG[df_BG['Location']=='TOP'].copy().reset_index(drop=True)


nan_rows = df_BG['MBC_mean'].isna()
df_BG=df_BG[~nan_rows].reset_index(drop=True)

df_BG['B-glucosidase_ug_pNP_per_g_per_hour'] = pd.to_numeric(df_BG['B-glucosidase_ug_pNP_per_g_per_hour'], errors='coerce')

# df_BG.select_dtypes(include=['object']).columns

obj_cols= ['B-glucosidase_ug_pNP_per_g_per_hour','SO4-S_mg_per_kg', 'B_mg_per_kg',
           'Zn_mg_per_kg', 'Mn_mg_per_kg', 'Cu_mg_per_kg', 'Fe_mg_per_kg']
for var in obj_cols:
    df_BG[var] = pd.to_numeric(df_BG[var], errors='coerce')

#%% run lambda model on all top core ICR data or load pre-computed data

# if run_lambda_model:
    
#     df = pd.read_csv('1000Soil_data/icr_v2_corems2.csv')
#     df = df.rename(columns={'Unnamed: 0': 'MolForm'})
#     CHEMICAL_ELEMENTS = ["C", "H", "N", "O", "P", "S"]
#     df[CHEMICAL_ELEMENTS] = df[CHEMICAL_ELEMENTS].fillna(0)
    
#     # Initial Columns as a list object
#     init_columns = ['MolForm', 'C', 'H', 'O', 'S', 'N', 'P', 'Index', 'Calculated m/z',
#            'H/C', 'O/C', 'Heteroatom Class','DBE']
    
#     # get soil sample names
#     samples = df.columns[13:137].values
#     # remove rows where all entries are 1
#     df_filtered = df[~(df[samples] == 1).all(axis=1)]
    
#     df_melt = pd.melt(df, id_vars=init_columns,
#                            value_vars=samples,
#                            var_name='Sample',
#                            value_name='Presence')
    
#     ph=df_BG[['Sample','pH']]
    
#     # top cores
#     df_top = df_melt[df_melt['Sample'].str[-3:] == 'TOP'].copy()
#     df_top = df_top.merge(ph, how='inner', on='Sample')
#     df_top_1 = df_top[df_top['Presence'] == 1].copy().reset_index(drop=True)
#     OC_count=df_top_1.groupby('Sample').count()
#     #calculate lambda
#     gt = get_compositions(df_top_1)
#     b1 = get_lambda(gt, 0,'TEEM')
#     teem_icr_df = pd.concat((df_top_1, b1), axis=1)
#     teem_icr_df.to_csv('ICR_top.csv.gz', compression='gzip', index=False)
    
# else:
#     #aggregate by each class and average lambda
#     ICR_top = pd.read_csv('ICR_top.csv.gz', compression='gzip')

# %% run lambda model on WEOM assigned clasees

def process_top_icr_data():

    icr_v2_corems2 = pd.read_csv('1000Soil_data/icr_v2_corems2.csv')
    icr_v2_corems2 = icr_v2_corems2.rename(columns={'Unnamed: 0': 'MolForm'})

    CHEMICAL_ELEMENTS = ["C", "H", "N", "O", "P", "S"]
    icr_v2_corems2[CHEMICAL_ELEMENTS] = icr_v2_corems2[CHEMICAL_ELEMENTS].fillna(0)
    
    icr_v2_corems2['Compound_Class'] = assign_class(icr_v2_corems2, "bs2")
    icr_v2_corems2['DBE'] = DBEcalc(icr_v2_corems2['C'], icr_v2_corems2['H'], icr_v2_corems2['N'])
    icr_v2_corems2['AImod'] = AImodcalc(icr_v2_corems2['C'], icr_v2_corems2['H'], icr_v2_corems2['N'], icr_v2_corems2['O'], icr_v2_corems2['S'])
   
    init_columns = ['MolForm','Calculated m/z', 'C', 'H', 'O', 'S', 'N', 'P', 'H/C', 'O/C',
                     'Compound_Class','AImod','DBE']

    # get sample (spectrum) names
    samples = icr_v2_corems2.columns[13:137].values
    # remove rows where all entries are 1
    df_filtered = icr_v2_corems2[~(icr_v2_corems2[samples] == 1).all(axis=1)]
    df_melt = pd.melt(df_filtered, id_vars=init_columns,
                      value_vars=samples,
                      var_name='Sample',
                      value_name='Presence')
    df_top = df_melt[df_melt['Sample'].str[-3:] == 'TOP'].copy()

    ph = df_BG[['Sample','Location', 'pH']]

    df_top = df_top.merge(ph, how='inner', on='Sample')
    df_top_1 = df_top[df_top['Presence'] == 1].copy().reset_index(drop=True)

    # OC_count = df_top_1.groupby('Sample').count()

    tcol = ['Compound_Class','Calculated m/z', 'C', 'H', 'O', 'N', 'S', 'P', 'H/C', 'O/C','AImod','DBE']

    samples = df_top_1['Sample'].unique()
    sample = 'SJER_TOP'

    df_by_class_final = pd.DataFrame()
    for sample in tqdm(samples):

        WEOC = df_BG.loc[df_BG['Sample'] == sample, "WEOC_mean"].values[0]
        subset_df = df_top_1[df_top_1['Sample'] == sample].reset_index(drop=True)
        rel_abud = subset_df['Compound_Class'].value_counts().reset_index()
        
        # remove 'Lignin','Cond Hydrocarbon','Tannin' as they may not be bioavailable for observed respiration
        # major_pool_rows = rel_abud['Compound_Class'].isin(['Lignin','Cond Hydrocarbon','Tannin'])
        # rel_abud = rel_abud[~major_pool_rows].reset_index(drop=True)
        frac_refractory_pool_size=0
        rel_abud['Rel. abundance g/gWOEM'] = rel_abud['count']/rel_abud['count'].sum()
        shannon_index = shannon_diversity_index(rel_abud['count'].values)
        
        df_by_class = subset_df[tcol].groupby('Compound_Class').mean().reset_index()
        df_by_class['Sample'] = sample
        df_by_class['Location'] = df_BG.loc[df_BG['Sample'] == sample, "Location"].values[0]

        df_by_class['pH'] = df_BG.loc[df_BG['Sample'] == sample, "pH"].values[0]
        df_by_class['shannon_diversity_index'] = shannon_index

        df_by_class = pd.merge(df_by_class, rel_abud, how='right', on="Compound_Class")
        temp = df_by_class['Rel. abundance g/gWOEM']*df_by_class['C']
        df_by_class['Rel. abundance gC/gC WOEM'] = temp/temp.sum()

        df_by_class['bioavail_WEOC mgC/gsoil'] = WEOC*(1-frac_refractory_pool_size)

        df_by_class['bioavail_OC conc mgC/gsoil'] = df_by_class['Rel. abundance gC/gC WOEM']*WEOC*(1-frac_refractory_pool_size)

        df_by_class_final = pd.concat((df_by_class_final, df_by_class))

    gt = get_compositions(df_by_class_final)
    b1 = get_lambda(gt, 0, 'TEEM')

    df_by_class_final = df_by_class_final.reset_index(drop=True)
    icr_by_class = pd.concat((df_by_class_final, b1), axis=1)
    return icr_by_class, df_by_class_final, df_top_1

icr_by_class_raw, df_by_class_final, df_top_1 = process_top_icr_data()


icr_by_class_raw['DR']=4-icr_by_class_raw['NOSC']
icr_by_class_raw['Site'] = icr_by_class_raw['Sample'].str.split('_').str[0]
icr_by_class_raw['abs_stoichMet_donor'] = icr_by_class_raw['stoichMet_donor'].abs()
icr_by_class_raw['qs_max'] = 24*3/(icr_by_class_raw['C']*icr_by_class_raw['DR']) # per day
icr_by_class_raw['mu_max'] = icr_by_class_raw['qs_max']*icr_by_class_raw['CUE']

icr_by_class_raw['dGcat kJ/mol electron'] = icr_by_class_raw['delGcat']/(icr_by_class_raw['C']*icr_by_class_raw['DR'])
# icr_by_class_raw['mu_max_alt'] = (3*(-icr_by_class_raw['dGcat kJ/mol electron'])-4.5)/200
icr_by_class_raw[['Compound_Class','CUE', 'qs_max','mu_max']].head(9)

biom=df_BG[["BiomeType","Sample"]]
icr_by_class_raw =  pd.merge(icr_by_class_raw, biom, how='left', on="Sample")

icr_by_class_raw = icr_by_class_raw.rename(columns={'count': 'alpha_diversity'})
#%%

t_col = ['Sample','Location','Compound_Class', 'C', 'NOSC','CUE', 'lambda','AImod','DBE','Calculated m/z',
         'Rel. abundance gC/gC WOEM','bioavail_OC conc mgC/gsoil','bioavail_WEOC mgC/gsoil',
         'alpha_diversity','abs_stoichMet_donor', 'stoichMet_hco3',
         'qs_max','mu_max','DR','BiomeType']

icr_by_class = icr_by_class_raw[icr_by_class_raw['Location'] == 'TOP'][t_col].copy().reset_index(drop=True)
icr_by_class.to_csv('processed_data/icr_by_class.csv', index=False)

#%% 

df_BG_ICR = df_BG[df_BG.Location=="TOP"].copy()
t_col = ['Sample','NOSC','CUE', 'lambda','AImod','DBE','Calculated m/z']

ICR_top_mean = icr_by_class[t_col].groupby('Sample').mean().reset_index()
df_BG_ICR = df_BG_ICR.merge(ICR_top_mean, how='inner', on='Sample')  # inner: use intersection of keys from both frames

therm_std = icr_by_class[t_col].groupby('Sample').std().reset_index()
therm_CV = pd.DataFrame()

for col in ['CUE', 'lambda']:
    therm_CV[f'{col}_CV'] = therm_std[f'{col}'] / ICR_top_mean[f'{col}']

therm_CV['Sample'] = ICR_top_mean['Sample']

df_BG_ICR = df_BG_ICR.merge(therm_CV, how='inner', on='Sample')  # inner: use intersection of keys from both frames


#% further processing of top ICR data categorized by class ===========================================================

samples = icr_by_class_raw['Sample'].unique()
diversity_df =[]
for sample in samples:

    rel_abud = icr_by_class_raw.loc[icr_by_class_raw['Sample'] == sample, ['Compound_Class', 'alpha_diversity']]
    shannon_index = shannon_diversity_index(rel_abud['alpha_diversity'].values)
    major_pool_rows = rel_abud['Compound_Class'].isin(['Lignin-like', 'Cond Hydrocarbon-like', 'Tannin-like'])
    labile_to_recal_Ratio = rel_abud.loc[~major_pool_rows, 'alpha_diversity'].sum() / rel_abud.loc[major_pool_rows, 'alpha_diversity'].sum()
    
    # Calculate the fraction of all unique compounds in Compound_Class
    unique_compound_fraction = dict(zip(rel_abud['Compound_Class'], rel_abud['alpha_diversity'] / rel_abud['alpha_diversity'].sum()))
    
    cue = icr_by_class_raw.loc[icr_by_class_raw['Sample'] == sample, "CUE"].values.reshape(len(rel_abud), 1)
    cue_rao = rao_quadratic_entropy(rel_abud['alpha_diversity'].values, cue)

    MW = icr_by_class_raw.loc[icr_by_class_raw['Sample'] == sample, 'Calculated m/z'].values.reshape(len(rel_abud), 1)
    MW_rao = rao_quadratic_entropy(rel_abud['alpha_diversity'].values, MW)
    
    nosc = icr_by_class_raw.loc[icr_by_class_raw['Sample'] == sample, 'NOSC'].values.reshape(len(rel_abud), 1)
    nosc_rao = rao_quadratic_entropy(rel_abud['alpha_diversity'].values, nosc)

    lambda_ = icr_by_class_raw.loc[icr_by_class_raw['Sample'] == sample, 'lambda'].values.reshape(len(rel_abud), 1)
    lambda_rao = rao_quadratic_entropy(rel_abud['alpha_diversity'].values, lambda_)
    
    # Append the results for the current sample to the DataFrame
    diversity_entry = {compound+"_frac": unique_compound_fraction.get(compound, 0) for compound in unique_compound_fraction.keys()}
    diversity_entry.update({
        'Sample': sample,
        'Shannon_Index': shannon_index,
        'labile_to_recal_Ratio': labile_to_recal_Ratio
    })   
    diversity_df.append(diversity_entry)
    
diversity_df = pd.DataFrame(diversity_df)


df_BG_ICR['Respiration']=df_BG_ICR['Respiration_24_h_ug_CO2-C_per_g_soil_per_day']/1000

df_BG_ICR = df_BG_ICR.merge(diversity_df, how='inner', on='Sample')  # inner: use intersection of keys from both frames

alpha_diversity = icr_by_class_raw[['Sample','alpha_diversity']].groupby('Sample').sum().reset_index()
df_BG_ICR = df_BG_ICR.merge(alpha_diversity, how='inner', on='Sample')  # inner: use intersection of keys from both frames

t_col = ['Sample','abs_stoichMet_donor','stoichMet_hco3','qs_max','mu_max']
therm_mean = icr_by_class_raw[t_col].groupby('Sample').mean().reset_index()
df_BG_ICR = df_BG_ICR.merge(therm_mean, how='inner', on='Sample')  # inner: use intersection of keys from both frames


#% data clean up===========================================================
df_BG_ICR = df_BG_ICR.drop(columns=['B-glucosidase_ug_pNP_per_g_per_hour',
       'Respiration_24_h_ug_CO2-C_per_g_soil_per_day'])

cols_to_drop = df_BG_ICR.columns[df_BG_ICR.columns.str.contains('median|standard_deviation|standard_error|_sd|_se')]
df_BG_ICR = df_BG_ICR.drop(columns=cols_to_drop)
df_BG_ICR = df_BG_ICR.rename(columns={
    'Respiration_mg_CO2-C_per_g_soil_per_day': 'Respiration'
    # 'B-glucosidase_ug_pNP_per_g_per_hour': 'B-glucosidase',
})
df_BG_ICR['CN_WEOM'] = df_BG_ICR.WEOC_mean/df_BG_ICR.WETN_mean
df_BG_ICR.to_csv('processed_data/df_BG_ICR.csv', index=False)


# %% Figure 1
# Geolocations Figure 1A
df_BG_ICR['Robs_norm'] = df_BG_ICR['Respiration']/df_BG_ICR['Respiration'].max()

temp = df_BG_ICR[~df_BG_ICR['Lat'].isna()].copy()
temp = temp[['Site', 'Lat', 'Long', 'BiomeType', 'Robs_norm']].reset_index(drop=True)
temp = temp.sort_values(by='Site')
min_size, max_size = 10, 50  # Define the min and max sizes for markers
norm_Robs_norm = (temp['Robs_norm'] - temp['Robs_norm'].min()) / (temp['Robs_norm'].max() - temp['Robs_norm'].min())
temp['size'] = norm_Robs_norm * (max_size - min_size) + min_size


fig = px.scatter_geo(temp, lat='Lat', lon='Long', color="BiomeType",
                     hover_name="Site",  # column added to hover information
                     size='size'  # size of markers
                     #   text=temp['Site']
                     )

fig.update_traces(
    textposition='top center',
    mode='markers+text',
    textfont=dict(size=16)
)
fig.update_layout(
    title='MONet sites',
    geo_scope='usa',
    width=900,  # Set the figure width
    height=620,   # Set the figure height
    margin={"r": 0, "t": 0, "l": 0, "b": 0},  # Set the margins to reduce white space
    legend=dict(
        x=0.55,  # X coordinate for the legend position (center)
        y=0,  # Y coordinate for the legend position (bottom)
        xanchor='center',  # Anchor the legend's x position to the center
        yanchor='top',  # Anchor the legend's y position to the top
        orientation='h',  # Horizontal orientation
        bgcolor="rgba(255, 255, 255, 0.8)",  # Background color with transparency
        bordercolor="Black",  # Border color
        borderwidth=1,  # Border width
        font=dict(
            size=17  # Set the font size of legend text
        )
    )
)

fig.show()
fig.write_html('figs/Figure1A.html', auto_open=False)

# Respiration_by_BiomeType_horizontal Figure 1B
# Use the same colormap as the Plotly geolocation map
biome_colors = px.colors.qualitative.Plotly  # Extract Plotly's qualitative color palette
biome_palette = {biome: biome_colors[i % len(biome_colors)] for i, biome in enumerate(df_BG_ICR['BiomeType'].unique())}
plt.figure(figsize=(6,4))
ax = sns.barplot(data=df_BG_ICR, x='Respiration', y='BiomeType', palette=biome_palette, orient='h')
plt.ylabel("", fontsize=14)
plt.xlabel(r'Respiration [$\mathrm{m}$g C-CO$_2$ g$^{-1}$ soil day$^{-1}$]', fontsize=14)
plt.title('(B)', loc='left', fontsize=16)
ax.tick_params(axis='y', labelsize=12)  # Set y-tick font size to 14
plt.tight_layout()
plt.savefig("figs/Figure1B.svg", dpi=300, bbox_inches='tight')




from scipy.stats import linregress

# Melt the dataframe
melted = df_BG_ICR.melt(id_vars=["alpha_diversity", "BiomeType"], 
                        value_vars=["Respiration", "mu_max", "CUE"])

# Set up the FacetGrid
g = sns.FacetGrid(
    data=melted,
    col="variable", hue="BiomeType", col_wrap=1, sharey=False,
    height=2.2, aspect=2.5, palette=biome_palette
)

# Draw regression plots
g.map(sns.regplot, "alpha_diversity", "value", scatter=True, 
      scatter_kws={'alpha': 1, 's': 20}, line_kws={'linewidth': 1.5})

# Compute p-values
pval_dict = {}
for var in melted["variable"].unique():
    subdata = melted[melted["variable"] == var]
    pval_dict[var] = {}
    for biome in subdata["BiomeType"].unique():
        group = subdata[subdata["BiomeType"] == biome]
        x = group["alpha_diversity"]
        y = group["value"]
        if len(x.dropna()) > 1 and len(y.dropna()) > 1:
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            pval_dict[var][biome] = p_value
        else:
            pval_dict[var][biome] = np.nan

# Rebuild legends with p-values
for ax, var in zip(g.axes.flat, melted["variable"].unique()):
    handles, _ = ax.get_legend_handles_labels()
    new_labels = []
    for biome in g.hue_names:
        p = pval_dict[var].get(biome, np.nan)
        if np.isnan(p):
            label = f"p = NA"
        else:
            label = f"p = {p:.2g}"
        new_labels.append(label)

    # Remove existing legend and add a new one
    ax.legend(handles, new_labels, title="", fontsize=9, title_fontsize=9, loc='upper right')


g.set_xlabels("Alpha Diversity",fontsize=12)

yvar_str = [r"Observed respiration" "\n" r"[$\mathrm{m}$g C-CO$_2$ g$^{-1}$ soil day$^{-1}$]",
            r"$\overline{\mu}_{max} \ [\mathrm{day^{-1}}]$", r"$\overline{\mathrm{CUE}} \ [-]$"]
# Set individual x-axis labels as the column names (i.e., "mu_max", "qs_max", "CUE")
title_Str=["(C)","(D)","(E)"]
for ax, col_name,tstr in zip(g.axes.flat, yvar_str,title_Str):
    ax.set_ylabel(col_name, fontsize=11)  # Set x-axis label to the column name
    ax.set_title(tstr, loc='left',fontsize=12)
    ax.tick_params(axis='both', labelsize=10)

# Move the legend outside of the third pane
g.set_titles('')
plt.tight_layout()
plt.savefig('figs/Figure1C.png', dpi=300, bbox_inches='tight')
plt.savefig('figs/Figure1C.svg', bbox_inches='tight')




# %% Figure 2

icr_top=icr_by_class.loc[icr_by_class['Location']=='TOP'].copy().reset_index(drop=True)
icr_top['Site'] = icr_top['Sample'].str.split('_').str[0]

df = icr_top[['Site', "Compound_Class", 'Rel. abundance gC/gC WOEM']].reset_index(drop=True)
y_pivoted = df.pivot(index="Compound_Class", columns="Site", values='Rel. abundance gC/gC WOEM')


pretty_palette = sns.color_palette("coolwarm", n_colors=len(icr_top['Site'].unique()))
sns.set_theme(style="whitegrid", font_scale=1.5)


fig, ax = plt.subplots(3, 1, figsize=(11, 10), sharex=True, 
                       gridspec_kw={'height_ratios': [2, 1, 1]})  # Adjust heights

ax[0].set_title("(A)", loc='left')
ax[1].set_title("(B)", loc='left')
ax[2].set_title("(C)", loc='left')


pretty_palette = sns.color_palette("muted", n_colors=len(y_pivoted.index.unique()))

y_pivoted.T.plot(kind='bar', stacked=True, alpha=0.9, ax=ax[0], color=pretty_palette)
ax[0].tick_params(axis='x', rotation=90, labelsize=10)
ax[0].set_ylabel(r"Rel. proportion [$\mathrm{gC \ gC^{-1} DOM}$]")
ax[0].set_xlabel("Sites")
ax[0].legend("", frameon=False)
ax[0].grid('off')

pretty_palette = sns.color_palette("coolwarm", n_colors=len(icr_top['Site'].unique()))

sns.boxplot(x="Site", y="mu_max", data=icr_top, hue="Site", palette=pretty_palette, ax=ax[1], width=0.5, legend=False,
            flierprops=dict(marker='d', markerfacecolor='grey', markersize=2.5))
sns.lineplot(df_BG_ICR, x="Site", y='mu_max', linewidth=1.5, color='k',  ax=ax[1])
ax[1].set_ylabel(r"$\mu_{max}$ $\mathrm{[day^{-1}]}$")
ax[1].set_ylim(0,1.5)
ax[1].set_yticklabels(ax[1].get_yticklabels(), rotation=90) 

ax[1].legend("", frameon=False)


sns.boxplot(x="Site", y="CUE", data=icr_top, hue="Site", palette=pretty_palette, ax=ax[2], width=0.5, legend=False,
            flierprops=dict(marker='d', markerfacecolor='grey', markersize=2.5))
sns.lineplot(df_BG_ICR, x="Site", y='CUE', linewidth=1.5, color='k',  ax=ax[2])
ax[2].tick_params(axis='x',which='both', rotation=90, labelsize=12)
ax[2].set_ylabel(r"CUE")
ax[2].set_ylim(0.2,0.7)
ax[2].set_xlabel("Sites", fontsize=16)
ax[2].legend("", frameon=False)

# Add legend at the bottom of the figure
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.525, -0.09), ncol=4, fontsize=14, frameon=True)

plt.tight_layout(h_pad=0.5)
plt.savefig("figs/Figure2.svg", bbox_inches="tight")
plt.savefig("figs/Figure2.png", dpi=300, bbox_inches="tight")


#%% kinetic analysis on top soil cores


df_BG_ICR_filter = df_BG_ICR[['Sample','abs_stoichMet_donor','WEOC_mean','stoichMet_hco3','mu_max','CUE'
                              ,'qs_max','Respiration','alpha_diversity'
                              ]].copy()

ydata_var ='Respiration'
ydata = df_BG_ICR_filter[ydata_var].values

def plot_data_mode(df_BG_ICR, ymodel_var,ydata_var):
    r2 = r2_score(df_BG_ICR[ydata_var], df_BG_ICR[ymodel_var])
    rmse = np.sqrt(mean_squared_error(df_BG_ICR[ydata_var], df_BG_ICR[ymodel_var]))
    
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    sns.scatterplot(data=df_BG_ICR, x=ymodel_var, y=ydata_var,
                    label=f"{ymodel_var} ($R^2$={r2:.2f}, rmse={rmse:.3f})")
    # plt.scatter(ytrue,ypred,label=f"{ylab} ($R^2$={r2:.2f}, rmse={rmse:.2f})", s=50,
    #             edgecolors='white', alpha=0.95)
    plt.plot([0.0, 1], [0.0, 1], linestyle='--', color='red', label="")
    plt.xlabel(r'Simulated $R$ [$\mathrm{m}$g C-CO$_2$ g$^{-1}$ soil day$^{-1}$]')
    plt.ylabel(r'Observed $R$ [$\mathrm{m}$g C-CO$_2$ g$^{-1}$ soil day$^{-1}$]')
    plt.legend(fontsize=12)
    
    plt.subplot(1,2,2)
    sns.lineplot(df_BG_ICR,  x='WEOC_mean', y=ymodel_var, color='red', label="Model")
    sns.scatterplot(df_BG_ICR,  x='WEOC_mean', y=ydata_var,label="observations")
    plt.xlabel(r"DOM concentration [$\mathrm{mgC \ g^{-1} soil}$]")
    plt.ylabel(r'Respiration [$\mathrm{m}$g C-CO$_2$ g$^{-1}$ soil day$^{-1}$]')
    plt.legend(fontsize=12)
    plt.tight_layout()
    

# =============================================================================
# Monod
# =============================================================================

def func_Monod(x, Km,Vmax):
    oc= x[:,0]
    # cue = x[:,1]
    # B = x[:,2]
    # return oc/(oc + Km)
    return Vmax*oc/(oc + Km)


xdata = df_BG_ICR_filter[['WEOC_mean']].values
popt, pcov = curve_fit(func_Monod, xdata, ydata, (0.05, 0.07), bounds=(0, [10, 200]), method='trf')
# Get the standard deviation (uncertainty) of the parameters
perr = np.sqrt(np.diag(pcov))

Km, Vmax = popt
df_BG_ICR_filter['R_Monod'] = func_Monod(xdata, Km,Vmax)

yvar = ['R_Monod']
ylab = "$R_{Monod}$"

plot_data_mode(df_BG_ICR_filter, 'R_Monod',ydata_var)


###########################################################################
icr_df = icr_by_class.copy()

icr_df=icr_by_class.loc[icr_by_class['Location']=='TOP'].copy().reset_index(drop=True)

xdata = icr_df['abs_stoichMet_donor'].values
# =============================================================================


# =============================================================================
# using MFA to parametrize growth function
# =============================================================================
def func_mean_true(x, vH,N):
    yoc = x[:, 0]
    oc = x[:, 1]
    yco2 = x[:, 2]
    mumax = x[:, 3]
    return N*mumax*yco2*np.exp(-yoc/(vH*oc))


x = df_BG_ICR_filter[['abs_stoichMet_donor','WEOC_mean','stoichMet_hco3', 'mu_max']].values

ydata = df_BG_ICR_filter[ydata_var].values
popt, pcov = curve_fit(func_mean_true, x, ydata, [0.5,1], bounds=(0, [10,10]), method='trf')
vH,N = popt

ymodel = func_mean_true(x,vH,N)
print(r2_score(ydata, ymodel))

df_BG_ICR_filter['R_MTS_mean'] = ymodel
plot_data_mode(df_BG_ICR_filter, 'R_MTS_mean',ydata_var)


#% data model 1:1 scatter plots


# =============================================================================
# using true sum respiration to parametrize respiration function
# =============================================================================


def MTS_func(icr_df, vH,N):
    mumax = icr_df['mu_max']
    mu =mumax*np.exp(-icr_df['abs_stoichMet_donor'] /(vH*icr_df['bioavail_OC conc mgC/gsoil']))
    # mu = np.exp(-icr_df['abs_stoichMet_donor'] /(vH*icr_df['bioavail_OC conc mgC/gsoil']))  # mu_by_mumax
    icr_df['R_N_2'] = icr_df['stoichMet_hco3'] * mu
    new_df = icr_df[['Sample', 'R_N_2']].groupby('Sample').sum().reset_index()
    R =  N*new_df['R_N_2'].values
    return R

def func_MTS(x, vH,N):
    return MTS_func(icr_df, vH,N)


popt, pcov = curve_fit(func_MTS, x, ydata, [0.1,0.1], bounds=(0, [1000,10]), method='trf')
vH,N = popt

ymodel = func_MTS(x,vH,N)

ydata = df_BG_ICR_filter[ydata_var].values
print(r2_score(ydata, ymodel))

mumax = icr_df['mu_max']
mu =mumax*np.exp(-icr_df['abs_stoichMet_donor'] /(vH*icr_df['bioavail_OC conc mgC/gsoil']))
# mu = np.exp(-icr_df['abs_stoichMet_donor'] /(vH*icr_df['bioavail_OC conc mgC/gsoil']))  # mu_by_mumax
icr_df['Rsum_MTS'] = N* icr_df['stoichMet_hco3'] * mu

df_BG_ICR_filter['Rsum_MTS']=ymodel

plot_data_mode(df_BG_ICR_filter, 'Rsum_MTS',ydata_var)


# =============================================================================
# Plotting
#% plot kinetics

r2_R_MTS_mean = r2_score(df_BG_ICR_filter[ydata_var],df_BG_ICR_filter["R_MTS_mean"])
r2_MTS = r2_score(df_BG_ICR_filter[ydata_var],df_BG_ICR_filter["Rsum_MTS"])
r2_monod = r2_score(df_BG_ICR_filter[ydata_var],df_BG_ICR_filter['R_Monod'] )
xt = np.linspace(0.01,1, 150)
R_Monod = Vmax*xt/(xt+Km)
yoc = x[:, 0]
yco2 = x[:, 2]
R_MTS_mean = N*mumax.mean()*yco2.mean()*np.exp(-yoc.mean()/(vH*xt))
cols= sns.color_palette("colorblind", n_colors=3)

sns.set(style="whitegrid", font_scale=1.5)
alpha=0.6
msz=50
plt.figure(figsize=(7, 6))
sns.scatterplot(data=df_BG_ICR_filter, x="WEOC_mean", y=ydata_var, s=msz, color="grey",
                label='Obs. respiration', alpha=alpha, edgecolors='black')

plt.plot(xt, R_Monod, color=cols[2], alpha=1, linewidth=2)
sns.scatterplot(data=df_BG_ICR_filter, x="WEOC_mean", y="R_Monod", s=msz, label=f'$R_{{Monod}}$ ($R^2$={r2_monod:.2f})',
                color=cols[2], alpha=alpha, edgecolors='black')

sns.lineplot(data=df_BG_ICR_filter, x="WEOC_mean", y="R_MTS_mean", color=cols[1], linewidth=2)
sns.scatterplot(data=df_BG_ICR_filter, x="WEOC_mean", y="R_MTS_mean", s=msz, label=f'$R_{{MTS}}^{{mean}}$ ($R^2$={r2_R_MTS_mean:.2f})',
                color=cols[1], alpha=alpha, edgecolors='black')

temp = df_BG_ICR_filter[['WEOC_mean', 'Rsum_MTS']].sort_values(by='WEOC_mean')
sns.lineplot(x='WEOC_mean', y='Rsum_MTS', data=temp, color=cols[0], linewidth=2)
sns.scatterplot(data=df_BG_ICR_filter, x="WEOC_mean", y="Rsum_MTS", s=msz, label=f'$R_{{MTS}}^{{multi}}$ ($R^2$={r2_MTS:.2f})',
                color=cols[0], alpha=alpha, edgecolors='black')

plt.xlabel(r"DOM concentration [$\mathrm{mgC \ g^{-1} soil}$]", fontsize=18)
plt.ylabel(r'Respiration [$\mathrm{m}$g C-CO$_2$ g$^{-1}$ soil day$^{-1}$]', fontsize=18)
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig("figs/Figure5.svg")
plt.savefig("figs/Figure5.png")

# =============================================================================
yvar = ['R_Monod','R_MTS_mean','Rsum_MTS']
yvar_lable = [r"$R_{Monod}$",r"$R_{{MTS}}^{{mean}}$",r"$R_{{MTS}}^{{multi}}$"]

tstr = "ABC"
fig, ax = plt.subplots(1,3,figsize=(12, 4.5), sharey=True)
for i, vzip in enumerate(zip(yvar, yvar_lable)):
    var, ylab =vzip
    r2 = r2_score(df_BG_ICR_filter[ydata_var], df_BG_ICR_filter[var])
    rmse = np.sqrt(mean_squared_error(df_BG_ICR_filter[ydata_var], df_BG_ICR_filter[var]))
    sns.scatterplot(data=df_BG_ICR_filter, x=var, y=ydata_var,
                    label="", alpha=1,ax=ax[i])
    ax[i].annotate(f"$R^2$={r2:.2f},\nrmse={rmse:.3f}", xy=(0.56,0.35), fontsize=12)
    ax[i].plot([0.0, .750], [0.0, .750], linestyle='--', color='red', label="")
    ax[i].tick_params(axis='both', labelsize=12)
    ax[i].set_xlabel(f'{ylab}')
    ax[i].set_title(f'({tstr[i]})', loc='left')
ax[0].set_ylabel(r'Observed $R$ [$\mathrm{m}$g C-CO$_2$ g$^{-1}$ soil day$^{-1}$]')
plt.tight_layout()

# plt.savefig("figs/Figure_A2_data_model_respiration.svg", dpi=300)
# plt.savefig("figs/Figure_A2_data_model_respiration.png", dpi=300)

#%%

# =============================================================================
r2_R_MTS_mean = r2_score(df_BG_ICR_filter[ydata_var],df_BG_ICR_filter["R_MTS_mean"])
r2_MTS = r2_score(df_BG_ICR_filter[ydata_var],df_BG_ICR_filter["Rsum_MTS"])
r2_monod = r2_score(df_BG_ICR_filter[ydata_var],df_BG_ICR_filter['R_Monod'] )
xt = np.linspace(0.01,1, 150)
R_Monod = Vmax*xt/(xt+Km)
yoc = x[:, 0]
yco2 = x[:, 2]
R_MTS_mean = N*mumax.mean()*yco2.mean()*np.exp(-yoc.mean()/(vH*xt))
cols= sns.color_palette("colorblind", n_colors=3)

sns.set(style="whitegrid", font_scale=1.5)
alpha=0.6
msz=50

plt.figure(figsize=(7, 6), dpi=300)
sns.scatterplot(data=df_BG_ICR_filter, x="WEOC_mean", y=ydata_var, s=msz, color="grey",
                label='Obs. respiration', alpha=alpha, edgecolors='black')
plt.plot(xt, R_Monod, color=cols[2], alpha=1, linewidth=2,label=f'$R_{{Monod}}$ ($R^2$={r2_monod:.2f})')
sns.lineplot(x='WEOC_mean', y='Rsum_MTS', data=temp, color=cols[0], linewidth=2,label=f'$R_{{MTS}}^{{mean}}$ ($R^2$={r2_R_MTS_mean:.2f})')
plt.xlabel(r"DOM concentration [$\mathrm{mgC \ g^{-1} soil}$]", fontsize=18)
plt.ylabel(r'Respiration [$\mathrm{m}$g C-CO$_2$ g$^{-1}$ soil day$^{-1}$]', fontsize=18)
plt.legend(fontsize=18)
plt.ylim(0,0.8);plt.xlim(0,1)
plt.tight_layout()








#%%

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor

# cols= ['Respiration','Total_Carbon_pct', 'WEOC_mean','pH', 'CN_WEOM',
#                'mean_GWC','alpha_diversity','BiomeType']
# col_to_pretty = {
#     "WEOC_mean": "DOM",
#     "CN_WEOM": "CN_ratio",
#     "mean_GWC": "Soil_moisture",
#     "alpha_diversity": "Alpha_diversity",
#     "BiomeType": "BiomeType"
# }


# # cols= ['Respiration','Total_Carbon_pct', 'WEOC_mean','CN_WEOM', 
# #                'mean_GWC','Clay_pct','pH','alpha_diversity',]

# df = df_BG_ICR[cols].copy()
# df = pd.get_dummies(df, columns=['BiomeType'],  prefix_sep=':',drop_first=True, dtype="bool")

# df.rename(columns=col_to_pretty, inplace=True)

# X, y = df.drop(['Respiration'], axis=1), df['Respiration']



# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# rf_model = RandomForestRegressor(n_estimators=100)

# rf_model.fit(X, y)
# y_pred = rf_model.predict(X)

# r2 = r2_score(y, rf_model.predict(X))
# print(f"R² Score: {r2:.4f}")


# fig, axes = plt.subplots(1,1, figsize=(8, 6))
# # Actual vs. Predicted Scatter Plot
# plt.scatter(y, y_pred, s=80)
# axes.plot([y.min(), y_pred.max()], [y.min(), y_pred.max()], color='red', ls='--')
# axes.set_xlabel("Observed",fontsize=20)
# axes.set_ylabel("Predicted", fontsize=20)
# plt.tick_params(axis='both', which='major', labelsize=14)
# plt.tight_layout()

# feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=True)

# plt.figure(figsize=(7 ,5))
# feature_importance.plot(kind='barh')
# plt.title("Feature Importance (Random Forest)")
# plt.show()


# top_features = feature_importance.nlargest(6).index
# print("Top 6 features:", top_features)

# import shap
# explainer = shap.Explainer(rf_model)
# shap_values = explainer(X)

# # # visualize the first prediction's explanation
# # shap.plots.waterfall(shap_values[10])

# # top_features_id = feature_importance.nlargest(8).index.tolist()  # Ensure it's a list
# # shap.plots.beeswarm(shap_values[:, top_features_id], plot_size=[6,6])

# fig, ax=plt.subplots(1,1)
# shap.plots.bar(shap_values, show=False, ax=ax)


# num_features = len(top_features)
# cols = 3  # Number of columns in the grid
# rows = int(np.ceil(num_features / cols) ) # Calculate the necessary number of rows

# fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))  # Adjust figsize as needed
# axes = axes.flatten()  # Flatten in case the layout is a 2D array
# str= "ABCDEFGHIJKLMN"
# import statsmodels.api as sm
# for i, feature in enumerate(top_features):
#     ax = axes[i]
#     shap.plots.scatter(shap_values[:, feature], ax=ax, show=False)
#     ax.tick_params(axis='both', which='major', labelsize=16)

#     idx = np.where(X.columns==feature)[0][0]
#     x = X.iloc[:,idx]
#     y_sv = shap_values[:,idx].values
#     lowess = sm.nonparametric.lowess(y_sv, x, frac=.5)
#     ax.plot(*list(zip(*lowess)), color="red", )

#     x_label = ax.get_xlabel()
#     y_label = ax.get_ylabel()
#     ax.set_xlabel(x_label, fontsize=16)
#     ax.set_ylabel("", fontsize=16)
#     ax.set_title("("+str[i]+")", fontsize=16)    
#     # ax.set_title(f"Dependence plot for {feature}")
# # fig.delaxes(axes[-1])
# plt.tight_layout()


