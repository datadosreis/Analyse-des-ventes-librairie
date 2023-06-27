import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import datetime

#Analyse des datas
def data_analyse(df):
    print(df.info())
    print("\n")
    print(df.describe())
    print("\n")
    print(df.isnull().sum())
    print("\n")
    print(df[df.duplicated()].head())

#Vérification de l'unicité de la clé primaire
def test_cle(df,columns):
    a=df.drop_duplicates(subset=columns).shape[0]
    b=df.shape[0]
    if a==b: print("La clé est unique")
    else: print("La clé n'est pas unique")


#Rechercher des outliers via méthode de la distance interquartile
def find_outliers(v):
    Q1 = np.quantile(v, 0.25)
    Q3 = np.quantile(v,0.75)
    EIQ = Q3 - Q1
    LI = round(Q1 - (EIQ*1.5),2)
    print("La limite inférieure est de",LI,)
    LS = round(Q3 + (EIQ*1.5),2)
    print("La limite supérieure est de",LS,)
    print("Liste des outliers ---")
    print(v.loc[(v < LI) | (v > LS)].to_frame())

#Rechercher des outliers via méthode du zscore
outliers=[]
def find_outliers_zscore(data):
    threshold=3
    mean=np.mean(data)
    std=np.std(data)
    
    for i in data:
        z_score=(i-mean) / std
        if np.abs(z_score)>threshold:
            outliers.append(i)
    return outliers

#Etat des jointures
def etat_jointure(df,columns):
    return df.value_counts(columns)

#Analyse des prix
def pricing_analysis(data):
    print(data.describe())
    print("\n")
    print(data.mode())
    print("\n")
    print("Skewness empirique =",round(data.skew(),2),)
    print("\n")
    plt.figure(figsize=(16,5))
    plt.hist(data,ec="black")
    plt.title("Distribution empirique des prix",fontsize=12)
    plt.xlabel("Prix", fontsize=10)
    plt.ylabel("Count", fontsize=10)
    plt.show()
    print("\n")
    plt.figure(figsize=(16,5))
    plt.boxplot(data, vert=False)
    plt.show()
    find_outliers(data)
    return

#Afficher les valeurs sur un pie chart
def make_autopct(values):
    def my_autopct(pct):
        total = sum (values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct


#Courbe de Lorenz & Indice de Gini
def lorenz_gini(data,title):
    x = data.values
    n = len(x)
    lorenz = np.cumsum(np.sort(x)) / x.sum()
    lorenz = np.append([0],lorenz)
    xaxis = np.linspace(0-1/n,1+1/n,n+1)
    plt.plot(xaxis,lorenz,drawstyle='steps-post')
    plt.plot([0,1], [0,1])
    plt.title(title)
    plt.show()
    AUC = (lorenz.sum() -lorenz[-1]/2 -lorenz[0]/2)/n
    S = 0.5 - AUC
    gini = 2*S
    print("Indice de Gini =",round(gini,2))
    return

#Mesure de tendance centrale
def tendance_centrale(data):
    print("Mesures de tendance centrale:")
    print("-"*20)
    print("Mode =",data.mode())
    print("Moyenne =",round(data.mean(),2))
    print("Médiane =",round(data.median(),2))

#Mesure de dispersion
def dispersion(data):
    print("Mesures de dispersion:")
    print("-"*20)
    print("Variance empirique =",round(data.var(),2))
    print("Variance empirique sans biais =",round(data.var(ddof=0),2))
    print("Ecart-Type empirique =",round(data.std(),2))
    print("Coéfficient de variation =",round(data.std(ddof=0),2))
    plt.figure(figsize=(16,5))
    plt.boxplot(data, vert=False)
    plt.show()
    return

#Mesure de forme
def forme(data):
    print("Mesures de forme:")
    print("-"*20)
    print("Skewness empirique =",round(data.skew(),2))
    print("Kurtosis empirique =",round(data.kurtosis(),2))

#Distribition de la variable
def distribution(data,title):
    plt.figure(figsize=(16,5))
    plt.hist(data,ec="black")
    plt.title(title,fontsize=12)
    plt.show()
    return

#Analyse univariée
def analyse_univariee(data,title_distribution,title_lorenz):
    plt.figure(figsize=(16,5))
    plt.hist(data,ec="black")
    plt.title(title_distribution,fontsize=12)
    plt.show()
    print("\n")
    print("Mesures de tendance centrale:")
    print("-"*20)
    print("Mode =",data.mode())
    print("Moyenne =",round(data.mean(),2))
    print("Médiane =",round(data.median(),2))
    print("\n")
    print("Mesures de dispersion:")
    print("-"*20)
    print("Variance empirique =",round(data.var(),2))
    print("Variance empirique sans biais =",round(data.var(ddof=0),2))
    print("Ecart-Type empirique =",round(data.std(),2))
    print("Coéfficient de variation =",round(data.std(ddof=0),2))
    plt.figure(figsize=(16,5))
    plt.boxplot(data, vert=False)
    plt.show()
    print("\n")
    print("Mesures de forme:")
    print("-"*20)
    print("Skewness empirique =",round(data.skew(),2))
    print("Kurtosis empirique =",round(data.kurtosis(),2))
    print("\n")
    print("Mesures de concentration:")
    print("-"*20)
    x = data.values
    n = len(x)
    lorenz = np.cumsum(np.sort(x)) / x.sum()
    lorenz = np.append([0],lorenz)
    xaxis = np.linspace(0-1/n,1+1/n,n+1)
    plt.plot(xaxis,lorenz,drawstyle='steps-post')
    plt.plot([0,1], [0,1])
    plt.title(title_lorenz)
    plt.show()
    AUC = (lorenz.sum() -lorenz[-1]/2 -lorenz[0]/2)/n
    S = 0.5 - AUC
    gini = 2*S
    print("Indice de Gini =",round(gini,2))
    return