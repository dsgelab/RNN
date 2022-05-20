
# A script to add Infectious,  SES, geographic registers in a longitudinal way 


```python
import pandas as pd
import gc
import time
import datetime as dt
import numpy as np
from tqdm import tqdm
```

# VACCINATION (will not use as most codes are in 2020-2021)


```python
vacc = pd.read_csv('/data/processed_data/thl_vaccination/vaccination_2022-01-14.csv', sep=";")
```


```python
vacc = vacc[vacc['LAAKEAINE'].notna()]
vacc['LAAKEAINE_5']=vacc['LAAKEAINE'].apply(lambda x: x[:5])

# some cleaning
vacc = vacc[vacc['LAAKEAINE_5']!='0']
vacc = vacc[vacc['LAAKEAINE_5']!='-1']
vacc = vacc[vacc['LAAKEAINE_5']!='-2']

print('Number of unique vaccination ATC codes before processing',vacc['LAAKEAINE'].nunique())
print('Number of unique vaccination ATC codes after processing',vacc['LAAKEAINE_5'].nunique())

vacc_ATC = vacc['LAAKEAINE_5'].unique().tolist()
```


```python
# checking whether vaccination ATC coesare contained within drug purchases ATC codes
dicty = pd.read_csv('/data/projects/project_avabalas/RNN/preprocessing/newcodes/code_dict_no_rare_additional_omits.csv')
purch_ATC = dicty[dicty['Source']=="D"]['Code'].values.tolist()
vacc_notin_purch = [item for item in vacc_ATC if item not in purch_ATC]
print(len(vacc_notin_purch), 'ATC codes are only in vaccination but not purchases register, and', len(vacc_ATC)-len(vacc_notin_purch), "are in both")
```


```python
# for longitudinal features date is necesary and unfortanatelly nearly 2/3 of the register is missing vaccination date, toseentires are removed
vacc = vacc[vacc['ROKOTE_ANTOPVM'].notna()]
```


```python
#remaining ATC codes when the date is removed
vacc_ATC2 = vacc['LAAKEAINE_5'].unique().tolist() 
vacc_notin_purch2 = [item for item in vacc_ATC2 if item not in purch_ATC]
print('After removing entries without a date',len(vacc_notin_purch2), 'ATC codes are only in vaccination but not purchases register, and', len(vacc_ATC2)-len(vacc_notin_purch2), "are in both")
```


```python
# add event age
dob = pd.read_csv('/data/processed_data/dvv/Finregistry_IDs_and_full_DOB.txt')
vacc['ROKOTE_ANTOPVM'] = pd.to_datetime(vacc['ROKOTE_ANTOPVM']) # diagnosis date
vacc = vacc.rename(columns={'TNRO': 'FINREGISTRYID'})
vacc = vacc.merge(dob, on='FINREGISTRYID', how='left')
vacc['DOB(YYYY-MM-DD)'] = pd.to_datetime(vacc['DOB(YYYY-MM-DD)'])  
vacc['EVENT_AGE'] = (vacc['ROKOTE_ANTOPVM'] - vacc['DOB(YYYY-MM-DD)']).dt.days/365.24
```


```python
#format columns
vacc = vacc.rename(columns={'ROKOTE_ANTOPVM': 'PVM', 'LAAKEAINE_5': 'CODE'})
vacc['SOURCE']="V"
vacc = vacc[['FINREGISTRYID', 'SOURCE', 'EVENT_AGE', 'PVM','CODE']]
```


```python
print('%_IDs',(vacc['FINREGISTRYID'].nunique()/7166416)*100,'N codes per ID', vacc.shape[0]/7166416,'unique codes after processing',vacc['CODE'].nunique() )
```


```python
print('These ATC codes had never a date recorded',[item for item in vacc_ATC if item not in vacc_ATC2])
```

# INFECTIOUS DISEASES


```python
inf_d = pd.read_csv('/data/processed_data/thl_infectious_diseases/infectious_diseases_2022-01-19.csv', sep=";")
```


```python
inf_d = inf_d[inf_d['reporting_group'].notna()]
inf_d = inf_d[inf_d['sampling_date'].notna()]
```


```python
# add event age
dob = pd.read_csv('/data/processed_data/dvv/Finregistry_IDs_and_full_DOB.txt')
inf_d['sampling_date'] = pd.to_datetime(inf_d['sampling_date']) # diagnosis date
inf_d = inf_d.rename(columns={'TNRO': 'FINREGISTRYID'})
inf_d = inf_d.merge(dob, on='FINREGISTRYID', how='left')
inf_d['DOB(YYYY-MM-DD)'] = pd.to_datetime(inf_d['DOB(YYYY-MM-DD)'])  
inf_d['EVENT_AGE'] = (inf_d['sampling_date'] - inf_d['DOB(YYYY-MM-DD)']).dt.days/365.24
```


```python
inf_d = inf_d.rename(columns={'sampling_date': 'PVM', 'reporting_group': 'CODE'})
inf_d['SOURCE']="ID"
inf_d = inf_d[['FINREGISTRYID', 'SOURCE', 'EVENT_AGE', 'PVM','CODE']]
```


```python
print('%_IDs',(inf_d['FINREGISTRYID'].nunique()/7166416)*100,'N codes per ID', inf_d.shape[0]/7166416,'unique codes after processing',inf_d['CODE'].nunique() )
```


```python
#replace Infectiois disease codes (which a lists of comma separated strings trasformed to a single string (such codes will cause issues down the line e.g. vhen using .split(",") )) with codes in format: IN_DIS_X
types={} 
for code in inf_d['CODE'].values.tolist():
    if code in types: continue
    else: types[code] = "IN_DIS_"+str(len(types)+1)

```


```python
# replace codes
code_list = inf_d['CODE'].values.tolist()
new_code_list = []
for code in code_list:
    if code in types: new_code_list.append(types[code])
    else: print(code, 'CODE NOT IN THE LIST!!!!')
inf_d['CODE'] = new_code_list
```


```python
types_df = pd.DataFrame([(k, v) for k, v in types.items()], columns=['Code','Token'])
print(types_df.shape[0])
types_df.to_csv('/data/projects/project_avabalas/RNN/preprocessing_new/Invectious_disease_code_dict.csv',index=False)
```

# SES


```python
import numpy as np
import matplotlib
matplotlib.use('Agg')
%matplotlib inline
from matplotlib import pyplot as plt

import pandas as pd
```


```python
inname = "/data/processed_data/sf_socioeconomic/sose_u1442_a.csv.finreg_IDsp"
df = pd.read_csv(inname,dtype={'psose':str,'sose':str})
```


```python
print('Unique psose values', df[df['psose'].notna()]['FINREGISTRYID'].nunique(),'unique sose values', df[df['sose'].notna()]['FINREGISTRYID'].nunique() )
```


```python
#create histogram of unique codes in psose
ax = df[df['vuosi']<=1985]['psose'].value_counts(dropna=False).plot(kind='bar')
ax.set_xlabel("unique psose codes (1970-1985)")
ax.set_ylabel("count")
plt.tight_layout()
plt.show()

#create histogram of unique sose codes between 1990-1993
ax = df[(df['vuosi']>=1990) & (df['vuosi']<=1993)]['sose'].value_counts(dropna=False).plot(kind='bar')
ax.set_xlabel("unique sose codes (1990-1993)")
ax.set_ylabel("count")
plt.tight_layout()
plt.show()

#create histogram of unique sose codes after 1995
ax = df[df['vuosi']>=1995]['sose'].value_counts(dropna=False).plot(kind='bar')
ax.set_xlabel("unique sose codes (1995-)")
ax.set_ylabel("count")
plt.tight_layout()
plt.show()
```


```python
#Now use the proposed harmonisation to convert the first two digits of psose codes to harmonised codes
psose_harm = {'11':'1','12':'1','21':'1','22':'1','31':'3','32':'3','33':'3','34':'3','41':'4','42':'4','43':'4','44':'4','51':'5','52':'5','53':'5','54':'5','7':'6','70':'6','6':'7','91':'8','92':'8','93':'8','99':'9'}
#NOTE: this follows the coding in "Socioeconomic status in FinRegistry google doc"
```


```python
#create histogram of unique codes in psose, when converted to harmonized codes
ax = df[df['vuosi']<=1985]['psose'].str[:2].replace(psose_harm).value_counts(dropna=False).plot(kind='bar')
ax.set_xlabel("unique harmonized psose codes (1970-1985)")
ax.set_ylabel("count")
plt.tight_layout()
```


```python
#Note that here we have assumed that the following codes that are not
#included in the statistics Finland definitions of sose codes for 1990-1993 are:
#10 = 1 = self-employed (all other codes starting with 1 are in this category)
#20 = 2 = self-employed (all other codes starting with 2 are in this category)
#70 = 7 = pensioners (all other codes starting with 7 are in this category)
sose1990_harm = {'10':'1','11':'1','12':'1','20':'1','21':'1','22':'1','23':'1','24':'1','29':'1','31':'3','32':'3','33':'3','34':'3','39':'3','41':'4','42':'4','43':'4','44':'4','49':'4','51':'5','52':'5','53':'5','54':'5','59':'5','6':'6','60':'6','7':'7','70':'7','71':'7','72':'7','73':'7','74':'7','79':'7','81':'8','82':'8','99':'9','91':'8','92':'8','93':'8'}
```


```python
#create histogram of unique sose codes between 1990-1993, when converted to harmonized codes
ax = df[(df['vuosi']>=1990) & (df['vuosi']<=1993)]['sose'].str[:2].replace(sose1990_harm).value_counts(dropna=False).plot(kind='bar')
ax.set_xlabel("unique harmonized sose codes (1990-1993)")
ax.set_ylabel("count")
plt.tight_layout()
```


```python
#Note that here we assume that code 99 stands for unknown as in the previous coding systems
#because code X is actually not present in the data even though it is supposed to mark unknown socioeconomic status
sose1995_harm = {'10':'1','11':'1','12':'1','20':'1','21':'1','22':'1','23':'1','24':'1','29':'1','31':'3','32':'3','33':'3','34':'3','39':'3','41':'4','42':'4','43':'4','44':'4','49':'4','51':'5','52':'5','53':'5','54':'5','59':'5','6':'6','60':'6','7':'7','70':'7','71':'7','72':'7','73':'7','74':'7','79':'7','81':'8','82':'8','99':'9'}
```


```python
#create histogram of unique sose codes after 1995, when converted to harmonized codes
ax = df[df['vuosi']>=1995]['sose'].str[:2].replace(sose1995_harm).value_counts(dropna=False).plot(kind='bar')
ax.set_xlabel("unique harmonized sose codes (1995-)")
ax.set_ylabel("count")
plt.tight_layout()
```


```python
df.loc[df['vuosi']<=1985,'psose'] = df[df['vuosi']<=1985]['psose'].str[:2].replace(psose_harm) # convert to harmonized codes psose codes between 1970-1985
```


```python
df.loc[(df['vuosi']>=1990) & (df['vuosi']<=1993),'sose'] = df[(df['vuosi']>=1990) & (df['vuosi']<=1993)]['sose'].str[:2].replace(sose1990_harm) # convert to harmonized codes sose codes between 1990-1993
```


```python
df.loc[df['vuosi']>=1995, 'sose'] = df[df['vuosi']>=1995]['sose'].str[:2].replace(sose1995_harm) # convert to harmonized codes sose codes after 1995
```


```python
#Combine psose and sose
df_sose = df[df['sose'].notna()].copy()
df_psose = df[df['psose'].notna()].copy()
df_sose = df_sose.drop(columns=['psose'])
df_psose = df_psose.drop(columns=['sose'])
df_psose = df_psose.rename(columns={'psose': 'sose'})
ses = pd.concat([df_psose, df_sose])
ses = ses.sort_values(["FINREGISTRYID", "vuosi"], ascending = (True, True))
```


```python
#Harmonized historgam
ax = ses['sose'].value_counts(dropna=False).plot(kind='bar')
ax.set_xlabel("unique harmonized sose codes (1970-)")
ax.set_ylabel("count")
plt.tight_layout()
```


```python
# add event age
dob = pd.read_csv('/data/processed_data/dvv/Finregistry_IDs_and_full_DOB.txt')
ses = ses.merge(dob, on='FINREGISTRYID', how='left')
ses['vuosi'] = ses['vuosi'].astype('int')
ses['vuosi'] = ses['vuosi'].astype('str')
ses['vuosi'] = ses['vuosi'].apply(lambda x: x+'-07-01')
ses['vuosi'] = pd.to_datetime(ses['vuosi']) 
ses['DOB(YYYY-MM-DD)'] = pd.to_datetime(ses['DOB(YYYY-MM-DD)'])  
ses['EVENT_AGE'] = (ses['vuosi'] - ses['DOB(YYYY-MM-DD)']).dt.days/365.24
```


```python
# check when in life IDs get ses recorded (many until fairly late in life) and other statistics
lines = []
total = ses['FINREGISTRYID'].nunique()
grouped1=ses.groupby(ses['FINREGISTRYID'])
for g,f in tqdm(grouped1, total=total):
    t_until_entry = f['EVENT_AGE'].iloc[0]
    unique_entries = f['sose'].nunique()
    history_length = float(f['EVENT_AGE'].iloc[-1:]-f['EVENT_AGE'].iloc[0])    
    line = [g,t_until_entry,unique_entries,history_length] #' '.join(sources),
    lines.append(line)
grouped1 = pd.DataFrame(lines,columns = ["FINREGISTRYID", "t_until_entry",'unique_entries','history_length'])
#for g, f in ass_all.groupby(ass_all['TNRO'])  
```


```python
print('Mean duration from birth until first SES entry',grouped1['t_until_entry'].mean(),'Number of different codes per individual',grouped1['unique_entries'].mean(),'mean follow-up duration form first to the last code', grouped1['history_length'].mean())
```


```python
ses = ses.rename(columns={'vuosi': 'PVM', 'sose': 'CODE'})
ses['SOURCE']="SES"
ses = ses[['FINREGISTRYID', 'SOURCE', 'EVENT_AGE', 'PVM','CODE']]
```


```python
ses['CODE'] = ses['CODE'].apply(lambda x: "SES_"+str(x))
```


```python
print('%_IDs',(ses['FINREGISTRYID'].nunique()/7166416)*100,'N codes per ID', ses.shape[0]/7166416,'unique codes after processing',ses['CODE'].nunique() )
```

# OCCUPATION


```python
# Ammaticodi https://www2.tilastokeskus.fi/fi/luokitukset/ammatti/ammatti_17_20210101/
# pamko is recorded only for years 1970/1975/1980/1985
# census description https://www.ilo.org/ilostat-files/SSM/SSM5/E/FI.html
# on average one ID has 9.3 'ammattikoodi'
# Pamko https://taika.stat.fi/fi/aineistokuvaus.html#!?dataid=FOLK_19701985_jua_vl7085_004.xml
```


```python
occupation = pd.read_csv('/data/processed_data/sf_socioeconomic/ammatti_u1442_a.csv.finreg_IDsp')
```


```python
amati = occupation[occupation['ammattikoodi'].notna()].copy()
```


```python
# add event age
amati['vuosi'] = amati['vuosi'].astype('int')
amati['vuosi'] = amati['vuosi'].astype('str')
amati['vuosi'] = amati['vuosi'].apply(lambda x: x+'-07-01')
amati['vuosi'] = pd.to_datetime(amati['vuosi'])
amati = amati.merge(dob, on='FINREGISTRYID', how='left')
amati['DOB(YYYY-MM-DD)'] = pd.to_datetime(amati['DOB(YYYY-MM-DD)'])  
amati['EVENT_AGE'] = (amati['vuosi'] - amati['DOB(YYYY-MM-DD)']).dt.days/365.24
```


```python
amati = amati.rename(columns={'vuosi': 'PVM', 'ammattikoodi': 'CODE'})
amati['SOURCE']="O_A"
amati = amati[['FINREGISTRYID', 'SOURCE', 'EVENT_AGE', 'PVM','CODE']]
```


```python
amati['CODE'] = amati['CODE'].apply(lambda x : x[:1]) # take just a first digit
```


```python
amati = amati.sort_values(["FINREGISTRYID", "EVENT_AGE"], ascending = (True, True))
```


```python
amati = amati.loc[(amati['PVM'] >= '1995-01-01')].copy() # because coddig appears to be different before 1995
```


```python
amati['CODE'] = amati['CODE'].apply(lambda x: "O_A_"+str(x))
```


```python
# check some statistics
lines = []
total = amati['FINREGISTRYID'].nunique()
grouped=amati.groupby(amati['FINREGISTRYID'])
for g,f in tqdm(grouped, total=total):
    unique_entries = f['CODE'].nunique()
    history_leangth = float(f['EVENT_AGE'].iloc[-1:]-f['EVENT_AGE'].iloc[0])
    line = [g,unique_entries,history_leangth] #' '.join(sources),
    lines.append(line)
grouped = pd.DataFrame(lines,columns = ["FINREGISTRYID","unique_entries", "history_leangth"])
#for g, f in ass_all.groupby(ass_all['TNRO'])  
```


```python
print('Number of different codes per individual',grouped['unique_entries'].mean(),'mean follow-up duration form first to the last code', grouped['history_leangth'].mean())
```


```python
print('%_IDs',(amati['FINREGISTRYID'].nunique()/7166416)*100,'N codes per ID', amati.shape[0]/7166416,'unique codes after processing',amati['CODE'].nunique() )
```

# EDUCATION


```python
education = pd.read_csv('/data/processed_data/sf_socioeconomic/tutkinto_u1442_a.csv.finreg_IDsp', encoding='latin-1')
# Educatiion level kaste_t2 naming https://www2.stat.fi/en/luokitukset/koulutusaste/koulutusaste_1_20160101/
# Education field: https://www2.stat.fi/en/luokitukset/koulutusala/
```


```python
education = education[education['vuosi'].notna()].copy()
```

## Education level


```python
# add event age
education['vuosi'] = education['vuosi'].astype('int')
education['vuosi'] = education['vuosi'].astype('str')
education['vuosi'] = education['vuosi'].apply(lambda x: x+'-07-01')
education['vuosi'] = pd.to_datetime(education['vuosi'])
education = education.merge(dob, on='FINREGISTRYID', how='left')
education['DOB(YYYY-MM-DD)'] = pd.to_datetime(education['DOB(YYYY-MM-DD)'])  
education['EVENT_AGE'] = (education['vuosi'] - education['DOB(YYYY-MM-DD)']).dt.days/365.24
```


```python
edu_l = education.copy()
edu_l = edu_l.rename(columns={'vuosi': 'PVM', 'kaste_t2': 'CODE'})
edu_l['SOURCE']="E_L"
edu_l = edu_l[['FINREGISTRYID', 'SOURCE', 'EVENT_AGE', 'PVM','CODE']]
```


```python
edu_l['CODE'] = edu_l['CODE'].apply(lambda x: "E_L_"+str(int(x)))
```


```python
print('%_IDs',(edu_l['FINREGISTRYID'].nunique()/7166416)*100,'N codes per ID', edu_l.shape[0]/7166416,'unique codes after processing',edu_l['CODE'].nunique() )
```

## Education field


```python
edu_f = education.copy()
edu_f = edu_f.rename(columns={'vuosi': 'PVM', 'iscfi2013': 'CODE'})
edu_f['SOURCE']="E_F"
edu_f = edu_f[['FINREGISTRYID', 'SOURCE', 'EVENT_AGE', 'PVM','CODE']]
```


```python
print('%_IDs',(edu_f['FINREGISTRYID'].nunique()/7166416)*100,'N codes per ID', edu_f.shape[0]/7166416,'unique codes after processing',edu_f['CODE'].nunique() )
```


```python
edu_f['CODE'] = edu_f['CODE'].apply(lambda x: "E_F_"+str(int(x)))
```


```python
# combine and save all longitudinal except GEO
longitudinal = pd.concat([inf_d, ses,amati,edu_l,edu_f],    # Combine vertically
                          ignore_index = True,
                          sort = False)
print(inf_d.shape[0]+ses.shape[0]+amati.shape[0]+edu_l.shape[0]+edu_f.shape[0]==longitudinal.shape[0])
```


```python
longitudinal.to_csv('/data/projects/project_avabalas/RNN/preprocessing_new/longitudinal_features.csv', index=False)
```

# GEOGRAPHIC


```python
geo = pd.read_csv('/data/projects/project_jgerman/living_pno_extended_2.csv', usecols=['FINREGISTRYID','Start_of_residence','End_of_residence','posti_alue','nimi','kunta'],dtype={'posti_alue': str})
```


```python
geo = geo.sort_values(["FINREGISTRYID", "Start_of_residence"], ascending = (True, True))
```


```python
geo = geo[geo['posti_alue'].notna()].copy()
```


```python
print('unique IDs', geo["FINREGISTRYID"].nunique(),'unique number of postal codes', geo['posti_alue'].nunique(),'unique number of munnicipality names',geo['nimi'].nunique(),'unique number of munnicipality codes',geo['kunta'].nunique())
```


```python
geo = geo[geo['Start_of_residence'].notna()].copy()
```


```python
dob = pd.read_csv('/data/processed_data/dvv/Finregistry_IDs_and_full_DOB.txt')
```


```python
geo['Start_of_residence'] = pd.to_datetime(geo['Start_of_residence'])
geo = geo.merge(dob, on='FINREGISTRYID', how='left')
geo['DOB(YYYY-MM-DD)'] = pd.to_datetime(geo['DOB(YYYY-MM-DD)'])  
geo['EVENT_AGE'] = (geo['Start_of_residence'] - geo['DOB(YYYY-MM-DD)']).dt.days/365.24
```


```python
geo = geo.rename(columns={'Start_of_residence': 'PVM', 'kunta': 'CODE'})
geo['SOURCE']="GEO"
geo = geo[['FINREGISTRYID', 'SOURCE', 'EVENT_AGE', 'PVM','CODE']]
```


```python
# check whibn in life IDs get ses recorded (many until fairly late in life) and other statistics
lines = []
total = geo['FINREGISTRYID'].nunique()
grouped2=geo.groupby(geo['FINREGISTRYID'])
for g,f in tqdm(grouped2, total=total):
    t_until_entry = f['EVENT_AGE'].iloc[0]
    unique_entries = f.shape[0]
    unique_municipalities = f['CODE'].nunique()
    history_length = float(f['EVENT_AGE'].iloc[-1:]-f['EVENT_AGE'].iloc[0])
    first_address_date = f['PVM'].iloc[0]
    last_address_date = f['PVM'].iloc[-1:].to_string().split()[1]
    line = [g,t_until_entry,unique_entries,unique_municipalities,history_length,first_address_date,last_address_date] #' '.join(sources),
    lines.append(line)
grouped2 = pd.DataFrame(lines,columns = ["FINREGISTRYID", "t_until_entry",'unique_entries','unique_municipalities','history_length','first_address_date','last_address_date'])
#for g, f in ass_all.groupby(ass_all['TNRO'])  
```


```python
print('Mean duration from birth until first geo entry',grouped2['t_until_entry'].mean(),'Number of  codes per individual',grouped2['unique_entries'].mean(),'Number of different municipalities per individual',grouped2['unique_municipalities'].mean(),'mean follow-up duration form first to the last code', grouped2['history_length'].mean())
```


```python
print('%_IDs',(geo['FINREGISTRYID'].nunique()/7166416)*100,'N codes per ID', geo.shape[0]/7166416,'unique codes after processing',geo['CODE'].nunique() ) 
```


```python
# filling in geo data of non-index children born after 2010 with their mothers geo data if availbale, if not with fathers. 
```


```python
relatives = pd.read_csv('/data/processed_data/dvv/Tulokset_1900-2010_tutkhenk_ja_sukulaiset.txt.finreg_IDsp',usecols=('FINREGISTRYID','Relationship','Relative_ID','Relative_DOB'))
relatives = relatives[relatives['Relative_ID'].notna()]
```


```python
dob_rel = dob.copy()
dob_rel['year']=dob_rel['DOB(YYYY-MM-DD)'].apply(lambda x: x[:4])
geo_allIDs2 = dob_rel.merge(geo, on='FINREGISTRYID', how='left') # add DOB to geographich information
geo_allIDs2['year'] = geo_allIDs2['year'].astype(int)
```


```python
children = geo_allIDs2[(geo_allIDs2['CODE'].isna())&(geo_allIDs2['year']>=2010)].copy() # filter only non index individuals who are born after 2010
children= children.drop(['SOURCE','EVENT_AGE','PVM','CODE'], axis = 1)
```


```python
min_p = pd.read_csv('/data/processed_data/minimal_phenotype/minimal_phenotype_2022-03-28.csv',usecols=('FINREGISTRYID','sex')) # read sex information
relatives = relatives.merge(min_p, on='FINREGISTRYID', how='left') # add sex infomration to relatives
relatives = relatives.rename(columns={'FINREGISTRYID': 'Parents_FINREGISTRYID','Relative_ID': 'FINREGISTRYID' })
```


```python
children2 = children.merge(relatives, on='FINREGISTRYID', how='left') # megre relative parent information to children dataframe
```


```python
# create a dataframe for ech non-index indivvidual with their fathers and mothers ID
lines = []
total = children2['FINREGISTRYID'].nunique()
grouped3=children2.groupby(children2['FINREGISTRYID'])
for g,f in tqdm(grouped3, total=total):
    mother = f[(f['sex']==1)&(f['Relationship']=='2')]['Parents_FINREGISTRYID']
    if mother.shape[0]>0:
        mother=mother.values[0]
    else: 
        mother=np.nan
    father = f[(f['sex']==0)&(f['Relationship']=='2')]['Parents_FINREGISTRYID']
    if father.shape[0]>0:
        father=father.values[0]
    else: 
        father=np.nan
    year = f['year'].iloc[0]
    line = [g,year,mother,father] #' '.join(sources),
    lines.append(line)
```


```python
grouped3 = pd.DataFrame(lines,columns = ["FINREGISTRYID", "B_year",'mother_ID','father_ID'])
grouped3 = grouped3.rename(columns={'FINREGISTRYID': 'childs_FINREGISTRYID','mother_ID': 'FINREGISTRYID' })
```


```python
chidrend_geo = grouped3.merge(geo, on='FINREGISTRYID', how='left') # Add mothers' GEO data to children borsn after 2010
chidrend_geo=chidrend_geo[chidrend_geo['CODE'].notna()].copy() # remove lines not containing mother indformatuon (it was not availbale for 21609 individuals)
chidrend_geo['event_year']=chidrend_geo['PVM'].dt.strftime('%Y')
chidrend_geo['event_year'] = chidrend_geo['event_year'].astype(int)
```


```python
chidrend_geo1 = chidrend_geo.iloc[:2617336,:].copy() # done in two loops as they slow down as appended DF becomes larger
```


```python
# for children born after 2010 create a dataframe with mothers geo information
cild_geo = pd.DataFrame()
total = chidrend_geo1['childs_FINREGISTRYID'].nunique()
grouped4=chidrend_geo1.groupby(chidrend_geo1['childs_FINREGISTRYID'])
for g,f in tqdm(grouped4, total=total):
    f.reset_index(drop=True, inplace=True)
    sub_df=f.iloc[-1:,:]
    if (sub_df['B_year']<sub_df['event_year']).item():
        a = f[f['event_year']>=f['B_year']].index[0]
        if a>0: a=a-1
        sub_df = f.iloc[a:,:]    
    cild_geo=cild_geo.append(sub_df)
```


```python
chidrend_geo2 = chidrend_geo.iloc[2617336:,:].copy()
```


```python
# for children born after 2010 create a dataframe with mothers geo information
cild_geo2 = pd.DataFrame()
total2 = chidrend_geo2['childs_FINREGISTRYID'].nunique()
grouped5=chidrend_geo2.groupby(chidrend_geo2['childs_FINREGISTRYID'])
for g,f in tqdm(grouped5, total=total):
    f.reset_index(drop=True, inplace=True)
    sub_df=f.iloc[-1:,:]
    if (sub_df['B_year']<sub_df['event_year']).item():
        a = f[f['event_year']>=f['B_year']].index[0]
        if a>0: a=a-1
        sub_df = f.iloc[a:,:]    
    cild_geo2=cild_geo2.append(sub_df)
```


```python
child_geo = cild_geo.append(cild_geo2)
print(child_geo['childs_FINREGISTRYID'].nunique(),geo['FINREGISTRYID'].nunique())
child_geo= child_geo.drop(['B_year','FINREGISTRYID','father_ID','event_year'], axis = 1)
child_geo = child_geo.rename(columns={'childs_FINREGISTRYID':'FINREGISTRYID'})
geo2 = geo.append(child_geo)
print(geo2['FINREGISTRYID'].nunique())
```


```python
geo2['CODE'] = geo2['CODE'].apply(lambda x: "GEO_"+str(int(x)))
```


```python
geo2.to_csv('/data/projects/project_avabalas/RNN/preprocessing_new/geo_features.csv', index=False)
```

# Combine datas


```python
all_long = pd.concat([longitudinal,geo2],    # Combine vertically
                     ignore_index = True,
                     sort = False)
```


```python
print(longitudinal.shape[0]+geo2.shape[0]==all_long.shape[0])
```


```python
all_long = all_long.sort_values(["FINREGISTRYID", "EVENT_AGE"], ascending = (True, True))
```


```python
all_long.to_csv('/data/projects/project_avabalas/RNN/preprocessing_new/all_long_features.csv', index=False)
```


```python
#all_long = pd.read_csv('/data/projects/project_avabalas/RNN/preprocessing_new/all_long_features.csv')
```


```python
all_long.reset_index(drop=True, inplace=True)
```


```python
# split large DF to chunks of 300k IDs
for ii in range(1,25):
    id = ii*300000+1
    if ii > 3:
        b = "FR"
    else: b = "FR0"
    id = b+str(id)
    print(id)
    if ii == 1:
        end_index = all_long[all_long['FINREGISTRYID']==id].index[0]
        sub_df = all_long.iloc[:end_index,:].copy()
    elif ii==24:
        sub_df = all_long.iloc[beginning_index:,:].copy()
    else:
        end_index = all_long[all_long['FINREGISTRYID']==id].index[0]
        sub_df = all_long.iloc[beginning_index:end_index,:].copy()
    
    sub_df.to_csv('/data/projects/project_avabalas/RNN/preprocessing_new/all_long_features.csv.'+str(ii), index=False)
    del sub_df
    gc.collect()
    
    beginning_index=end_index
        
```