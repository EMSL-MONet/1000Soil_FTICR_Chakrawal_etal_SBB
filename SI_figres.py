


# %%

from scipy import stats

from scipy.stats import f_oneway
from sklearn.model_selection import train_test_split
# from scipy.stats import randint, expon
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
# import matplotlib as mpl
from statsmodels.formula.api import ols
import plotly.express as px
from scipy.optimize import curve_fit

import numpy as np
import pandas as pd
sns.set_theme(font_scale=1.0, style='whitegrid')

#%%

icr_by_class = pd.read_csv('processed_data/icr_by_class.csv')

df = pd.read_csv('processed_data/df_BG_ICR.csv')

df['logRespiration']=np.log10(df['Respiration'])
df['DR']=4-df['NOSC']

list(df.columns)
df['CNB'] = df.MBC_mean/df.MBN_mean

df['CN_WEOM_WETN'] = df.WEOC_mean/df.WETN_mean

df['f_WEOM'] =df.WEOC_mean/ (df.Total_Carbon_pct*10)
df['Robs_norm'] = df['Respiration']/df['Respiration'].max()



#% Supplementarty Figures 

#%% scatter plots Relationship between variables describing the chemical nature of DOM, biogeochem and respiration
#% Supplementarty Figures S1 S2 S3

therm_indices = ['Shannon_Index', 'alpha_diversity', 'CUE_CV', 'lambda_CV']

y_var = 'Respiration'
corr, p_value = stats.pearsonr(df['pH'], df[y_var])
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6), sharey=True)
axes = axes.flatten()

for i, var in enumerate(therm_indices):
    ax = axes[i]
    subset = df[[var, y_var]].dropna()
    
    # Compute Pearson correlation
    corr, p_value = stats.pearsonr(subset[var], subset[y_var])
    color = 'red' if p_value < 0.05 else 'black'
    
    # Create scatter plot with regression line
    sns.regplot(data=subset, x=var, y=y_var, scatter=True, fit_reg=True, ci=None,
                scatter_kws={'s': 10}, ax=ax)
    ax.set_xlabel(var)
    ax.set_ylabel('')

    
    # Annotate correlation coefficient
    ax.text(0.05, 0.85, f'$r$ = {corr:.2f}', color=color, transform=ax.transAxes,
            fontsize=12, fontweight='normal', backgroundcolor='w')

# axes[-1].remove()
fig.supylabel(r'Respiration [log10 ($\mathrm{m}$g C-CO$_2$ g$^{-1}$ soil day$^{-1}$)]')

plt.tight_layout(w_pad=2)
plt.savefig("figs/SI/FigureS1.svg", dpi=300, bbox_inches='tight')


therm_indices = ['NOSC', 'CUE','lambda', 'AImod', 'DBE', 'Calculated m/z']

y_var = 'Respiration'

sns.set(style='whitegrid', font_scale=1)
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8, 8), sharey=True)
axes = axes.flatten()

for i, var in enumerate(therm_indices):
    ax = axes[i]
    subset = df[[var, y_var]].dropna()
    
    # Compute Pearson correlation
    corr, p_value = stats.pearsonr(subset[var], subset[y_var])
    color = 'red' if p_value < 0.05 else 'black'
    
    # Create scatter plot with regression line
    sns.regplot(data=subset, x=var, y=y_var, scatter=True, fit_reg=True, ci=None,
                scatter_kws={'s': 10}, ax=ax)
    ax.set_xlabel(var)
    ax.set_ylabel('')

    
    # Annotate correlation coefficient
    ax.text(0.05, 0.85, f'$r$ = {corr:.2f}', color=color, transform=ax.transAxes,
            fontsize=12, fontweight='normal', backgroundcolor='w')

# axes[-1].remove()
fig.supylabel(r'Respiration [log10 ($\mathrm{m}$g C-CO$_2$ g$^{-1}$ soil day$^{-1}$)]')
# fig.suptitle('Relationship between variables describing the chemical nature of DOM and respiration')
plt.tight_layout(w_pad=2)
plt.savefig("figs/SI/FigureS2.svg", bbox_inches='tight')




biogeochem_indices = ['Total_Nitrogen_pct', 'Total_Carbon_pct', 'C_to_N_ratio',
                 'WEOC_mean', 'WETN_mean', 'MBC_mean', 'MBN_mean', 'CNB', 'CN_WEOM_WETN', 'pH', 'Soil.Temperature.C','Clay_pct']

y_var = 'Respiration'

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(10, 8), sharey=True)
axes = axes.flatten()

for i, var in enumerate(biogeochem_indices):
    ax = axes[i]
    subset = df[[var, y_var]].dropna()
    
    # Compute Pearson correlation
    corr, p_value = stats.pearsonr(subset[var], subset[y_var])
    color = 'red' if p_value < 0.05 else 'black'
    
    # Create scatter plot with regression line
    sns.regplot(data=subset, x=var, y=y_var, scatter=True, fit_reg=True, ci=None,
                scatter_kws={'s': 10}, ax=ax)
    ax.set_xlabel(var, fontsize=10)
    ax.set_ylabel('')
    
    # Annotate correlation coefficient
    ax.text(0.05, 0.85, f'$r$ = {corr:.2f}', color=color, transform=ax.transAxes,
            fontsize=12, fontweight='normal', backgroundcolor='w')

fig.supylabel(r'Respiration [log10 ($\mathrm{m}$g C-CO$_2$ g$^{-1}$ soil day$^{-1}$)]')
plt.tight_layout()
plt.savefig("figs/SI/FigureS3.svg", bbox_inches='tight')


# %% feature selection Figure S4 and S5

df_stat = pd.read_csv('processed_data/df_BG_ICR.csv')
df_stat.rename(columns={'CN_WEOM': 'CN_WEOM_WETN'}, inplace=True)

print(df_stat.select_dtypes(include=['object']).columns)
t_df = df_stat.drop(columns=['Sample_ID', 'Site', 'Site_Code', 'Location', 'P_Extract','Site_Code'])
nan_columns = t_df.columns[t_df.isna().any()]
nan_counts = t_df.isna().sum()
nan_columns = nan_counts[nan_counts >0].index.tolist()
nan_columns.remove('Soil.Temperature.C')
print(nan_columns)

t_df = df_stat.drop(columns=nan_columns)

numerical_df = t_df.select_dtypes(include=['number'])
X, y = numerical_df.drop('Respiration', axis=1), numerical_df['Respiration']
# Standardization
X_scaled = (X - X.mean())/X.std()

BiomeType_encoded = pd.get_dummies(t_df['BiomeType'], columns=["BiomeType"], drop_first=True, dtype='float')
# assemble processed dataframe
df_proc = pd.concat((t_df["Respiration"], X_scaled, BiomeType_encoded), axis=1)
X, y = df_proc.drop('Respiration', axis=1), df_proc['Respiration']


# =============================================================================
# remove unsignificant predictors
# =============================================================================

corr_matrix = df_proc.corr(method='pearson')
p_values = df_proc.corr(method=lambda x, y: stats.pearsonr(
    x, y)[1]) - pd.DataFrame(np.eye(len(df_proc.columns)), columns=df_proc.columns, index=df_proc.columns)

temp_R = corr_matrix["Respiration"].reset_index()
temp_R['p_val'] = p_values["Respiration"].values

# % identify and drop low siginifiance in correlation
# %matplotlib widget

plt.figure(figsize=(12, 5.5))
sns.lineplot(data=p_values['Respiration'], marker='o', label="P-val")
sns.lineplot(data=corr_matrix['Respiration'], marker='o', label="pearson")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("figs/SI/FigureS4.svg", bbox_inches='tight')


threshold = 0.05
to_drop = list(p_values[p_values['Respiration'] > 0.05].index)

filtered_numerical_df = df_proc.drop(columns=to_drop)

corr_matrix = filtered_numerical_df.corr(method='pearson')

plt.figure(figsize=(12, 9))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1,
            linewidths=0.5, annot_kws={"fontsize": 6}, xticklabels=True, yticklabels=True)
# plt.title('Correlation Heatmap of Numerical Columns')
plt.xticks(rotation=90, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig("figs/SI/FigureS5.svg", bbox_inches='tight')

# %% Figure S6
icr_top=pd.read_csv('processed_data/icr_by_class.csv')
icr_top['Site'] = icr_top['Sample'].str.split('_').str[0]

pretty_palette = sns.color_palette("coolwarm", n_colors=len(icr_top['Site'].unique()))

fig, ax = plt.subplots(4,1,figsize=(9, 9), sharex=True)
ax=ax.flatten()
sns.boxplot(x="Site", y="abs_stoichMet_donor", data=icr_top, hue="Site", palette=pretty_palette, ax=ax[0], width=0.5, legend=False,
            flierprops=dict(marker='d', markerfacecolor='grey', markersize=2.5))
sns.lineplot(df, x="Site", y='abs_stoichMet_donor', linewidth=1.5, color='k',  ax=ax[0])
ax[0].tick_params(axis='x', rotation=90)
ax[0].set_ylabel(r"$y_{OC}$")
ax[0].set_ylim()

ax[0].legend("", frameon=False)

sns.boxplot(x="Site", y="stoichMet_hco3", data=icr_top, hue="Site", palette=pretty_palette, ax=ax[1], width=0.5, legend=False,
            flierprops=dict(marker='d', markerfacecolor='grey', markersize=2.5))
sns.lineplot(df, x="Site", y='stoichMet_hco3', linewidth=1.5, color='k',  ax=ax[1])
ax[1].tick_params(axis='x', rotation=90)
ax[1].set_ylabel(r"$y_{CO_2}$")
ax[1].legend("", frameon=False)

sns.barplot(df, x="Site", y='WEOC_mean', ax=ax[2])
ax[2].tick_params(axis='x', rotation=90)
ax[2].set_ylabel(r"DOM concentration" + "\n" +r"[$\mathrm{mgC \ g^{-1} soil}$]")
ax[2].legend("", frameon=False)


sns.barplot(x="Site", y='alpha_diversity', data=df)
ax[3].set_yscale('log')
ax[3].tick_params(direction='out')

ax[3].tick_params(axis='x', rotation=90, labelsize=10)
ax[3].set_ylabel("Alpha diversity")

ax[3].set_xlabel("Sites")
ax[3].grid('off')

ax[0].set_title("(A)",loc='left')
ax[1].set_title("(B)",loc='left')
ax[2].set_title("(C)",loc='left')
ax[3].set_title("(D)",loc='left')
plt.tight_layout()
plt.savefig("figs/SI/FigureS6.svg")



#%% FigureS7

df_BG_ICR_temp=df.copy().reset_index(drop=True)
res = stats.spearmanr(df_BG_ICR_temp.alpha_diversity, df_BG_ICR_temp.labile_to_recal_Ratio)
corr_value = res.statistic
p_val = res.pvalue

# Create the figure and the subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns

# First plot: Alpha diversity vs Labile to recalcitrant ratio
res1 = stats.spearmanr(df_BG_ICR_temp.alpha_diversity, df_BG_ICR_temp.labile_to_recal_Ratio)
corr_value1 = res1.statistic
p_val1 = res1.pvalue

sns.regplot(ax=axes[0], data=df_BG_ICR_temp, x="labile_to_recal_Ratio", y='alpha_diversity', 
            scatter_kws={'s': 40}, line_kws={'color': 'purple', 'linewidth': 2})
axes[0].annotate(f'Corr: {corr_value1:.2f} ($p$ ={p_val1:.3f})', 
                 xy=(0.05, 0.8), xycoords='axes fraction', fontsize=12, backgroundcolor='white')
axes[0].set_xlabel('Labile to recalcitrant ratio', fontsize=14)
axes[0].set_ylabel('Alpha diversity', fontsize=14)
axes[0].set_title('(A)', fontsize=14, loc='left')

# Second plot: CUE vs Labile to recalcitrant ratio
res2 = stats.spearmanr(df_BG_ICR_temp.CUE, df_BG_ICR_temp.labile_to_recal_Ratio)
corr_value2 = res2.statistic
p_val2 = res2.pvalue

sns.regplot(ax=axes[1], data=df_BG_ICR_temp, x="labile_to_recal_Ratio", y='CUE', 
            scatter_kws={'s': 40}, line_kws={'color': 'purple', 'linewidth': 2})
axes[1].annotate(f'Corr: {corr_value2:.2f} ($p$ ={p_val2:.2E})', 
                 xy=(0.05, 0.8), xycoords='axes fraction', fontsize=12, backgroundcolor='white')
axes[1].set_xlabel('Labile to recalcitrant ratio', fontsize=14)
axes[1].set_ylabel('Carbon use efficiency', fontsize=14)
axes[1].set_title('(B)', fontsize=14, loc='left')

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('figs/SI/FigureS7.svg', bbox_inches='tight')


#%% Figure S8
sns.set_theme(style="whitegrid", font_scale=1.0)


# Bin WEOC into categories based on whether values are less than or greater than the mean
df['WEOC_bins'] = pd.cut(df['WEOC_mean'], bins=[-float('inf'), 
                df['WEOC_mean'].mean(), float('inf')], labels=["Below Mean", "Above Mean"])


# Create subplots for DR, mu_max, and abs_stoichMet_donor
fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=False)


# Variables to plot
variables = ["mu_max", "stoichMet_hco3","abs_stoichMet_donor"]
ystr = [r"$\mu_{max} \ [\mathrm{d^{-1}}]$", r"$y_{CO_2}$",r"$y_{OC}$"]
for i, var in enumerate(variables):
    below_mean = df[df["WEOC_bins"] == "Below Mean"][var]
    above_mean = df[df["WEOC_bins"] == "Above Mean"][var]

    # Perform one-tailed t-test
    t_stat, p_value = stats.ttest_ind(below_mean, above_mean)

    # Create the boxplot
    sns.boxplot(data=df, x="WEOC_bins", y=var, palette="Set2", ax=axes[i],hue='WEOC_bins', legend=False)
    axes[i].axhline(y=df[var].mean(), color='red', linestyle='--', linewidth=1.5)
    axes[i].set_xlabel("DOM concentration")
    axes[i].set_ylabel(ystr[i])
    axes[i].set_title(f"\ntwo-tailed t-test p-value: {p_value:.3f}")

plt.tight_layout()
plt.savefig("figs/SI/FigureS8.svg", bbox_inches='tight')




#%% Figure S9 Create a grid for contour plotting 

# fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

# sns.regplot(data=icr_by_class, x="abs_stoichMet_donor", y='mu_max', ax=axes[0], scatter_kws={'s': 40})

# sns.regplot(data=icr_by_class, x="abs_stoichMet_donor", y='stoichMet_hco3', ax=axes[1], scatter_kws={'s': 40})

# plt.tight_layout()
# plt.show()

vh = 1
N=1
OC= icr_by_class['bioavail_OC conc mgC/gsoil']
yOC= icr_by_class['abs_stoichMet_donor']
mu_max = icr_by_class['mu_max']
yco2= icr_by_class['stoichMet_hco3']
resp_MTS = N*mu_max*yco2*np.exp(-yOC/vh*OC)
icr_by_class['resp_MTS'] = resp_MTS
icr_by_class['resp_norm']=icr_by_class['resp_MTS']/icr_by_class['resp_MTS'].max()

# sns.lmplot(icr_by_class, x="abs_stoichMet_donor", y='resp_MTS', hue=None, legend=False)
# sns.lmplot(icr_by_class, x="mu_max", y='resp_MTS', hue=None, legend=False)
# sns.lmplot(icr_by_class, x="mu_max", y='resp_MTS', hue=None, legend=False)

# sns.lmplot(icr_by_class, x="abs_stoichMet_donor", y='resp_norm', hue=None, legend=False)
# sns.lmplot(icr_by_class, x="stoichMet_hco3", y='resp_norm', hue=None, legend=False)
# sns.lmplot(icr_by_class, x="mu_max", y='resp_norm', hue=None, legend=False)


x = icr_by_class["abs_stoichMet_donor"]
y = icr_by_class["mu_max"]
z = icr_by_class["resp_norm"]
xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
xi, yi = np.meshgrid(xi, yi)
from scipy.interpolate import griddata

# Interpolate z onto the grid
zi = griddata((x, y), z, (xi, yi), method='linear')

# Create a contour plot using matplotlib
plt.figure(figsize=(6, 4.5))
contour = plt.contourf(xi, yi, zi, levels=10, cmap='viridis')
plt.colorbar(contour, label="Normalized Respiration Rate")
plt.xlabel(r"$y_{OC}$")
plt.ylabel(r"$\mu_{max}$ $[\mathrm{d^{-1}}]$")
plt.tight_layout()
plt.savefig("figs/SI/FigureS9.svg",bbox_inches='tight')


#%% Figure S10 
fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)
# Plot 1: mu_max vs WEOC_mean
sns.regplot(data=df, x=df["alpha_diversity"], y=df['WEOC_mean'], scatter_kws={'s': 40}, ax=axes[0])
corr1, pval1 = stats.pearsonr(df['alpha_diversity'].dropna(), df['WEOC_mean'])

axes[0].legend([f"r: {corr1:.2f}, p-val: {pval1:.2e}"], loc="upper left")
axes[0].set_ylabel(r"DOM [$\mathrm{mg \, C \, g^{-1} \, soil}$]")
axes[0].set_xlabel("Alpha Diversity")

# Plot 3: abs_stoichMet_donor vs WEOC_mean
sns.regplot(data=df, x=df["alpha_diversity"], y=df['Total_Carbon_pct'], scatter_kws={'s': 40}, ax=axes[1])
corr3, pval3 = stats.pearsonr(df['alpha_diversity'].dropna(), df['Total_Carbon_pct'].dropna())
axes[1].legend([f"r: {corr3:.2f}, p-val: {pval3:.2e}"], loc="upper left")
axes[1].set_ylabel(r"Total Carbon [%]")
axes[1].set_xlabel("Alpha Diversity")

plt.tight_layout()

plt.savefig("figs/SI/FigureS10.svg" ,bbox_inches='tight')
plt.show()



#%% Figure S11

sns.set_theme(style='whitegrid', font_scale=1.2)
fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)

# Plot 1: mu_max vs alpha_diversity
sns.regplot(data=df, y=np.log10(df["mu_max"]), x=np.log10(df['alpha_diversity']), scatter_kws={'s': 40}, ax=axes[0])
corr1, pval1 = stats.pearsonr(np.log10(df['alpha_diversity'].dropna()), np.log10(df['mu_max'].dropna()))
axes[0].legend([f"r: {corr1:.2f}, p-val: {pval1:.2e}"], loc="upper left")
axes[0].set_xlabel(r"log$_{10}$ (Alpha Diversity)")
axes[0].set_ylabel(r"log$_{10}$ $(\mu_{max}$)")

# Plot 3: abs_stoichMet_donor vs alpha_diversity
sns.regplot(data=df, y=np.log10(df["abs_stoichMet_donor"]), x=np.log10(df['alpha_diversity']), scatter_kws={'s': 40}, ax=axes[1])
corr3, pval3 = stats.pearsonr(np.log10(df['alpha_diversity'].dropna()), np.log10(df['abs_stoichMet_donor'].dropna()))
axes[1].legend([f"r: {corr3:.2f}, p-val: {pval3:.2e}"], loc="upper left")
axes[1].set_xlabel(r"log$_{10}$ (Alpha Diversity)")
axes[1].set_ylabel(r"log$_{10}$ ($y_{OC}$)")

plt.tight_layout()

plt.savefig("figs/SI/FigureS11.svg",bbox_inches='tight')
plt.show()

#%% Figure S12

x = df[['abs_stoichMet_donor','WEOC_mean','stoichMet_hco3', 'mu_max']].values

ydata = df['Respiration'].values

def func_mean_true(x, vH,N):
    yoc = x[:, 0]
    oc = x[:, 1]
    yco2 = x[:, 2]
    mumax = x[:, 3]
    return N*mumax*yco2*np.exp(-yoc/(vH*oc))


popt, pcov = curve_fit(func_mean_true, x, ydata, [0.5,1], bounds=(0, [10,10]), method='trf')
vH,N = popt

ymodel = func_mean_true(x,vH,N)
df['R_MTS_mean'] = ymodel

# Generate 100 realizations of respiration rate by varying yoc, yco2, and mumax

data = df[['abs_stoichMet_donor', 'stoichMet_hco3', 'mu_max']].values  # Convert to NumPy array
def func_Temp(x, yoc, yco2, mumax, vH, N):  # average of the parameters across soils
    return N * mumax * yco2 * np.exp(-yoc / (vH * x))

xt = np.linspace(0.01, 1, 150)

num_realizations = len(data)
realization_results = []
for row in range(num_realizations):
    yoc_sample, yco2_sample, mumax_sample = data[row, 0], data[row, 1], data[row, 2]
    realization_results.append(func_Temp(xt, yoc_sample, yco2_sample, mumax_sample, vH, N))


# Plot the realizations
# Convert results to a DataFrame for easier plotting
realization_df = pd.DataFrame(realization_results).T
realization_df['xt'] = xt


fig, ax = plt.subplots(figsize=(8, 5))
norm = plt.Normalize(vmin=df['alpha_diversity'].min(), vmax=df['alpha_diversity'].max())
sm = plt.cm.ScalarMappable(cmap='cool', norm=norm)
sm.set_array([])

for i, col in enumerate(realization_df.columns[:-1]):  # Exclude 'xt' column
    alpha_div_value = df.iloc[i]['alpha_diversity']
    ax.plot(realization_df['xt'], realization_df[col], linewidth=1.5,
            color=plt.cm.cool(norm(alpha_div_value)), alpha=0.5)

fig.colorbar(sm, ax=ax, label='Alpha Diversity')

# Plot the mean line
mean_values = realization_df.iloc[:, :-1].mean(axis=1)
plt.plot(realization_df['xt'], mean_values, color='black', linewidth=3, label=' Ensemble mean')
plt.scatter(df['WEOC_mean'], df['Respiration'], color='grey', alpha=0.5, label='Observation')
plt.scatter(df['WEOC_mean'], df['R_MTS_mean'], color='blue',zorder=3, alpha=0.5,label='Single pool MTS' )

# Add labels and legend
plt.xlabel(r"DOM [$\mathrm{mgC \ g^{-1} soil}$]", fontsize=16)
plt.ylabel(r'Respiration [$\mathrm{m}$g C-CO$_2$ g$^{-1}$ soil day$^{-1}$]', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()

plt.savefig("figs/SI/FigureS12.svg", bbox_inches='tight')






#%% alpha_diversity_by_compound_class NOT IN SI

sns.set_theme(style='whitegrid', font_scale=1.7)
icr_by_class['WEOC_bins'] = pd.cut(icr_by_class['bioavail_WEOC mgC/gsoil'], bins=[-float('inf'), 
                                icr_by_class['bioavail_WEOC mgC/gsoil'].mean(), float('inf')],
                                labels=["Below Mean", "Above Mean"])

plt.figure(figsize=(12, 7))
sns.boxplot(data=icr_by_class, x="Compound_Class", y="alpha_diversity", palette="Set2", hue="WEOC_bins")
plt.xlabel("")
plt.yscale('log')
plt.legend(title="DOM concentration", loc="upper right", fontsize=16, title_fontsize=16)
plt.ylabel("Alpha Diversity", fontsize=16)
# plt.ylim(0, 2000)
plt.xticks(rotation=90)  # Rotate x-axis labels to avoid overlap

# Perform t-tests for each Compound_Class group
compound_classes = icr_by_class["Compound_Class"].unique()
for i, compound_class in enumerate(compound_classes):
    below_mean_alpha = icr_by_class[(icr_by_class["Compound_Class"] == compound_class) & (icr_by_class["WEOC_bins"] == "Below Mean")]["alpha_diversity"]
    above_mean_alpha = icr_by_class[(icr_by_class["Compound_Class"] == compound_class) & (icr_by_class["WEOC_bins"] == "Above Mean")]["alpha_diversity"]
    
    # Perform one-tailed t-test
    t_stat, p_value = stats.ttest_ind(above_mean_alpha, below_mean_alpha, alternative='greater')
    
    # Add asterisks for significance above the max whisker
    max_whisker = icr_by_class[(icr_by_class["Compound_Class"] == compound_class) & (icr_by_class["WEOC_bins"] == "Above Mean")]["alpha_diversity"].max()
    if p_value < 0.05:
        plt.text(i, max_whisker + 50, '*', ha='center', va='bottom', fontsize=20, color='black')

plt.tight_layout()
# plt.savefig("figs/SI/alpha_diversity_by_compound_class.png", dpi=300, bbox_inches='tight')
# plt.savefig("figs/SI/alpha_diversity_by_compound_class.svg", dpi=300, bbox_inches='tight')



#%% This block creates a grid of subplots to visualize the relationship between alpha diversity 
# and various compound fractions (e.g., Lignin, Protein, Lipid, etc.) in the dataset.

sns.set_theme(style="whitegrid", font_scale=1.4)
fractions = [
    'Lignin-like_frac',
       'Cond Hydrocarbon-like_frac', 'Tannin-like_frac', 'Protein-like_frac',
       'Unsat Hydrocarbon-like_frac', 'Carbohydrate-like_frac',
       'Amino Sugar-like_frac', 'Lipid-like_frac', 'Other_frac'
]

# Create subplots
fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey=False)
axes = axes.flatten()
yvar = 'alpha_diversity'

# yvar = 'Respiration'

for i, fraction in enumerate(fractions):
    # Filter out rows with NaN or inf values in the relevant columns
    valid_data = df[[yvar, fraction]].replace([np.inf, -np.inf], np.nan).dropna()
    
    # Calculate Pearson correlation and p-value
    corr, pval = stats.pearsonr(valid_data[yvar], valid_data[fraction])
    
    sns.regplot(data=valid_data, x=yvar, y=fraction, ax=axes[i], scatter_kws={'s': 50}, line_kws={'color': 'red'})
    
    # Set ylabel for each subplot
    axes[i].set_ylabel(f"{fraction.replace('_frac', '').replace('_', ' ')}")
    
    # Annotate correlation with significance
    significance = "*" if pval < 0.05 else ""
    axes[i].annotate(f"r = {corr:.2f}{significance}", xy=(0.05, 0.85),
                      xycoords='axes fraction', fontsize=18, backgroundcolor='white')
    
    # Set xlabel only for the last row of subplots
    if i >= 6:  # Last three panels
        axes[i].set_xlabel(yvar)
    else:
        axes[i].set_xlabel("")

# Adjust layout
plt.tight_layout()
plt.show()


#%%


df['R_MBC_specific'] = df['Respiration']/df['MBC_mean']
df['Robs_norm'] = df['Respiration']/ df['Respiration'].max()

df_BG_ICR_filter = df[['Sample','abs_stoichMet_donor','WEOC_mean','stoichMet_hco3','mu_max','CUE'
                              ,'qs_max','Respiration','alpha_diversity',
                              'R_MBC_specific','Robs_norm']].copy()

ydata_var ='Respiration'
ydata = df_BG_ICR_filter[ydata_var].values

def plot_data_mode(df, ymodel_var,ydata_var):
    r2 = r2_score(df[ydata_var], df[ymodel_var])
    rmse = np.sqrt(mean_squared_error(df[ydata_var], df[ymodel_var]))
    
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    sns.scatterplot(data=df, x=ymodel_var, y=ydata_var,
                    label=f"{ymodel_var} ($R^2$={r2:.2f}, rmse={rmse:.3f})")
    # plt.scatter(ytrue,ypred,label=f"{ylab} ($R^2$={r2:.2f}, rmse={rmse:.2f})", s=50,
    #             edgecolors='white', alpha=0.95)
    plt.plot([0.0, 1], [0.0, 1], linestyle='--', color='red', label="")
    plt.xlabel(r'Simulated $R$ [$\mathrm{m}$g C-CO$_2$ g$^{-1}$ soil day$^{-1}$]')
    plt.ylabel(r'Observed $R$ [$\mathrm{m}$g C-CO$_2$ g$^{-1}$ soil day$^{-1}$]')
    plt.legend(fontsize=12)
    
    plt.subplot(1,2,2)
    sns.lineplot(df,  x='WEOC_mean', y=ymodel_var, color='red', label="Model")
    sns.scatterplot(df,  x='WEOC_mean', y=ydata_var,label="observations")
    plt.xlabel(r"DOM [$\mathrm{mgC \ g^{-1} soil}$]")
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


# Monod with variable Vmax acrosss soils from FTICR for peer review
def func_Monod(x, Km,N):
    oc= x[:,0]
    Vmax = x[:,1]
    cue = x[:,2]
    # B = x[:,2]
    # return oc/(oc + Km)
    return N*(1-cue)*Vmax*oc/(oc + Km)
x = df_BG_ICR_filter[['WEOC_mean', 'qs_max','CUE']].values
popt, pcov = curve_fit(func_Monod, x, ydata, (0.05, 0.07), bounds=(0, [10, 200]), method='trf')
# Get the standard deviation (uncertainty) of the parameters
perr = np.sqrt(np.diag(pcov))

Km2,N = popt
df_BG_ICR_filter['R_Monod2'] = func_Monod(x, Km2,N)


plot_data_mode(df_BG_ICR_filter, 'R_Monod2',ydata_var)



###########################################################################
icr_df = icr_by_class.copy()

icr_df=icr_by_class.loc[icr_by_class['Location']=='TOP'].copy().reset_index(drop=True)

xdata = icr_df['abs_stoichMet_donor'].values
# =============================================================================


def sum_func_monod(icr_df,  Km,N):
    Vmax = icr_df['qs_max']
    oc= icr_df['bioavail_OC conc mgC/gsoil']
    cue = icr_df['CUE']
    icr_df['R_monod_sum'] = (1-cue)*Vmax*oc/(oc + Km)
    new_df = icr_df[['Sample', 'R_monod_sum']].groupby('Sample').sum().reset_index()
    R =  N*new_df['R_monod_sum'].values
    return R

def fit_sum_func_monod(x, Km,N):
    return sum_func_monod(icr_df,Km,N)


popt, pcov = curve_fit(fit_sum_func_monod, xdata, ydata, [0.1,0.1], bounds=(0, [1000,10]), method='trf')
Km3,N = popt
ymodel = fit_sum_func_monod(xdata,Km3,N)

df_BG_ICR_filter['R_monod_sum'] = ymodel

plot_data_mode(df_BG_ICR_filter, 'R_monod_sum',ydata_var)


plt.figure(figsize=(11,5))

ymodel_var = 'R_Monod'
ylab = "fitted $V_{max}$"

r2 = r2_score(df_BG_ICR_filter[ydata_var], df_BG_ICR_filter[ymodel_var])
rmse = np.sqrt(mean_squared_error(df_BG_ICR_filter[ydata_var], df_BG_ICR_filter[ymodel_var]))

plt.subplot(1,2,1)
sns.scatterplot(data=df_BG_ICR_filter, x=ymodel_var, y=ydata_var,color='black',
                label=f"{ylab} ($R^2$={r2:.2f}, rmse={rmse:.3f})")

ymodel_var = 'R_Monod2'
ylab = "single pool $V_{max}$"
r2 = r2_score(df_BG_ICR_filter[ydata_var], df_BG_ICR_filter[ymodel_var])
rmse = np.sqrt(mean_squared_error(df_BG_ICR_filter[ydata_var], df_BG_ICR_filter[ymodel_var]))

sns.scatterplot(data=df_BG_ICR_filter, x=ymodel_var, y=ydata_var,color='grey',alpha=0.5, 
                label=f"{ylab} ($R^2$={r2:.2f}, rmse={rmse:.3f})")

ymodel_var = 'R_monod_sum'
ylab = "multi pool $V_{max}$"
r2 = r2_score(df_BG_ICR_filter[ydata_var], df_BG_ICR_filter[ymodel_var])
rmse = np.sqrt(mean_squared_error(df_BG_ICR_filter[ydata_var], df_BG_ICR_filter[ymodel_var]))

sns.scatterplot(data=df_BG_ICR_filter, x=ymodel_var, y=ydata_var,color='grey',alpha=0.5, 
                label=f"{ylab} ($R^2$={r2:.2f}, rmse={rmse:.3f})")


plt.plot([0.0, 1], [0.0, 1], linestyle='--', color='red', label="1:1")
plt.xlabel(r'Simulated $R$ [$\mathrm{m}$g C-CO$_2$ g$^{-1}$ soil day$^{-1}$]')
plt.ylabel(r'Observed $R$ [$\mathrm{m}$g C-CO$_2$ g$^{-1}$ soil day$^{-1}$]')
plt.legend(fontsize=12)



plt.subplot(1,2,2)
xt = np.linspace(0.01,1, 150)
R_Monod = Vmax*xt/(xt+Km)
plt.plot(xt,R_Monod, color='red',alpha=1,linewidth=2,label="fitted $V_{max}$")
sns.lineplot(df_BG_ICR_filter,  x='WEOC_mean', y='R_Monod2', color='green', label="single pool",linewidth=2)
sns.lineplot(df_BG_ICR_filter,  x='WEOC_mean', y='R_monod_sum', color='blue', label="multi pool",linewidth=2)

sns.scatterplot(df_BG_ICR_filter,  x='WEOC_mean', y=ydata_var,label="observations", color='grey')
plt.xlabel(r"DOM [$\mathrm{mgC \ g^{-1} soil}$]")
plt.ylabel(r'Respiration [$\mathrm{m}$g C-CO$_2$ g$^{-1}$ soil day$^{-1}$]')
plt.legend(fontsize=14)
plt.tight_layout()



#%%
def analyze_random_forest(data, xvars, target):
    """
    Perform Random Forest analysis, plot feature importance, and generate SHAP plots.

    Parameters:
    - data: DataFrame containing the dataset.
    - xvars: List of feature column names.
    - target: Target column name.

    Returns:
    - rf_model: Trained Random Forest model.
    - shap_values: SHAP values for the model.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt
    import pandas as pd
    import shap
    import statsmodels.api as sm
    import numpy as np
    from sklearn.inspection import permutation_importance

    # Prepare data
    X, y = data[xvars], data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)

    # Calculate R² scores
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    print(f"Training R² Score: {r2_train:.4f}")
    print(f"Testing R² Score: {r2_test:.4f}")

    # Plot Actual vs. Predicted for training and testing
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].scatter(y_train, y_train_pred, s=80, color='blue', alpha=0.6, label='Training')
    axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', ls='--')
    axes[0].set_xlabel(f"Observed {target} (Training)", fontsize=16)
    axes[0].set_ylabel(f"Predicted {target} (Training)", fontsize=16)
    axes[0].annotate(f"$R^2$ = {r2_train:.2f}", xy=(0.05, 0.9), xycoords='axes fraction', fontsize=14, color='blue')
    axes[0].legend(fontsize=12)

    axes[1].scatter(y_test, y_test_pred, s=80, color='green', alpha=0.6, label='Testing')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', ls='--')
    axes[1].set_xlabel(f"Observed {target} (Testing)", fontsize=16)
    axes[1].set_ylabel(f"Predicted {target} (Testing)", fontsize=16)
    axes[1].annotate(f"$R^2$ = {r2_test:.2f}", xy=(0.05, 0.9), xycoords='axes fraction', fontsize=14, color='green')
    axes[1].legend(fontsize=12)

    plt.tight_layout()

    # Feature importance
    feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=True)
    plt.figure(figsize=(7, 5))
    feature_importance.plot(kind='barh')
    plt.title("Feature Importance (Random Forest)")
    plt.show()

    # SHAP analysis
    explainer = shap.Explainer(rf_model, X_train)
    shap_values = explainer(X_train, check_additivity = False)

    shap.plots.beeswarm(shap_values)
    fig, ax = plt.subplots(1, 1, figsize=(3, 6))
    shap.plots.bar(shap_values, show=False, ax=ax)

    # Top features
    top_features = feature_importance.nlargest(6).index
    num_features = len(top_features)
    cols = 3  # Number of columns in the grid
    rows = int(np.ceil(num_features / cols))  # Calculate the necessary number of rows

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))  # Adjust figsize as needed
    axes = axes.flatten()  # Flatten in case the layout is a 2D array
    str_labels = "ABCDEFGHIJKLMN"
    for i, feature in enumerate(top_features):
        ax = axes[i]
        shap.plots.scatter(shap_values[:, feature], ax=ax, show=False)
        ax.tick_params(axis='both', which='major', labelsize=16)

        idx = np.where(X_train.columns == feature)[0][0]
        x = X_train.iloc[:, idx]
        y_sv = shap_values[:, idx].values
        lowess = sm.nonparametric.lowess(y_sv, x, frac=.5)
        ax.plot(*list(zip(*lowess)), color="red")

        x_label = ax.get_xlabel()
        ax.set_xlabel(x_label, fontsize=18)
        ax.set_ylabel("", fontsize=18)
        ax.set_title("(" + str_labels[i] + ")", fontsize=16)

    plt.tight_layout()

    # Compute permutation importance
    result = permutation_importance(rf_model, X_test, y_test, n_repeats=10, random_state=42)

    # Extract mean importance and standard deviation
    importance_mean = result.importances_mean
    importance_std = result.importances_std
    features = X_test.columns

    # Sort features by importance (descending order)
    sorted_idx = np.argsort(importance_mean)[::-1]

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance_mean[sorted_idx], y=np.array(features)[sorted_idx], xerr=importance_std[sorted_idx], orient="h", color="blue")
    plt.xlabel("Permutation Importance (Mean Decrease in Performance)")
    plt.ylabel("Features")
    plt.title("Feature Importance via Permutation")
    plt.show()

    return rf_model, shap_values

#%%

target = 'CUE'
xvars = ['WEOC_mean', 'abs_stoichMet_donor', 'stoichMet_hco3', 'mu_max']
biogeochem=['Total_Carbon_pct', 'WEOC_mean','pH', 'CN_WEOM_WETN',
               'mean_GWC']

cols = biogeochem + [target] + fractions
xvars=biogeochem+fractions
df1 = df[cols].copy()

rf_model, shap_values = analyze_random_forest(df1, xvars, target)


# #==============================================================================
# target = 'Respiration'

# xvars= ['Total_Carbon_pct', 'WEOC_mean','pH', 'CN_WEOM_WETN',
#                'mean_GWC','alpha_diversity','BiomeType']
# cols = xvars + [target]
# df = df[cols].copy()
# df = pd.get_dummies(df, columns=['BiomeType'],  prefix_sep=':',drop_first=True, dtype="int")

# X, y = df.drop(['Respiration'], axis=1), df['Respiration']
# xvars=X.columns.to_list()
# rf_model, shap_values = analyze_random_forest(df, xvars, target)


# #==============================================================================
# target = 'mu_max'
# xvars = ['WEOC_mean', 'alpha_diversity','NOSC']
# cols = xvars + [target]
# df = df[cols].copy()
# rf_model, shap_values =analyze_random_forest(df, xvars, target)

# target = 'abs_stoichMet_donor'
# xvars = ['WEOC_mean', 'alpha_diversity','NOSC']
# cols = xvars + [target]
# df = df[cols].copy()
# rf_model, shap_values =analyze_random_forest(df, xvars, target)




# xvars = ['bioavail_OC conc mgC/gsoil', 'alpha_diversity','DR']

# cols = xvars + [target]
# df = icr_by_class[cols].copy()
# rf_model, shap_values =analyze_random_forest(df, xvars, target)
# # cols= ['Rsum_MTS','bioavail_OC conc mgC/gsoil','abs_stoichMet_donor', 'stoichMet_hco3', 'mu_max']
# # df = icr_df[cols].copy()


# %%
