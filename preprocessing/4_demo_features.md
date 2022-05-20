

```python
import pandas as pd
import gc
import time
import datetime as dt
import numpy as np
from tqdm import tqdm

path = '/data/processed_data/minimal_phenotype/minimal_phenotype_2022-03-28.csv'
start_time = time.time()
df = pd.read_csv(path)
run_time = time.time()-start_time;print(run_time)
```

# Minimal phenotype


```python
#CONTINOUS

#df['longitude_last'] = df['longitude_last'].fillna(df['longitude_last'].mean(skipna=True))
#df['latitude_last'] = df['latitude_last'].fillna(df['latitude_last'].mean(skipna=True))
#df['latitude_first'] = df['latitude_first'].fillna(df['latitude_first'].mean(skipna=True))
#df['longitude_first'] = df['longitude_first'].fillna(df['longitude_first'].mean(skipna=True))


#df['residence_start_date_first'] = pd.to_datetime(df['residence_start_date_first'] )
#df['residence_end_date_first'] = pd.to_datetime(df['residence_end_date_first'] )
#df['residence_duration_first'] = (df['residence_end_date_first'] - df['residence_start_date_first']).dt.days/365.24
#df['residence_duration_first'] = df['residence_duration_first'].fillna(df['residence_duration_first'].mean(skipna=True))

df['Today'] = '2018-01-01'#date.today() 
df['Today'] = pd.to_datetime(df['Today'])
df['date_of_birth'] = pd.to_datetime(df['date_of_birth'])
df['AGE'] = (df['Today'] - df['date_of_birth']).dt.days/365.24
```


```python
continous = df[["FINREGISTRYID",'AGE']].copy()
```


```python
ordinal = df[["FINREGISTRYID",'number_of_children','drug_purchases','kanta_prescriptions']].copy()
```


```python
ordinal['number_of_children'] = ordinal['number_of_children'].fillna(ordinal['number_of_children'].mean(skipna=True))
ordinal['drug_purchases'] = ordinal['drug_purchases'].fillna(ordinal['drug_purchases'].mean(skipna=True))
ordinal['kanta_prescriptions'] = ordinal['kanta_prescriptions'].fillna(ordinal['kanta_prescriptions'].mean(skipna=True))
```


```python
binary = df[["FINREGISTRYID",'sex','in_social_assistance_registries','in_social_hilmo','index_person',
         'birth_registry_mother','birth_registry_child','in_vaccination_registry','in_infect_dis_registry','in_malformations_registry','in_cancer_registry']].copy()
```


```python
#df['residence_type_last']=df['residence_type_last'].fillna(4)
#one_hot = pd.get_dummies(df['residence_type_last'])
#one_hot = one_hot.rename(columns={1.0: 'residence_with_DVV',2.0: 'residence_no_DVV',3.0: 'residence_foreign',4.0: 'residence_nan'})
#one_h = df[["FINREGISTRYID"]].join(one_hot.copy())

#df['residence_type_first']=df['residence_type_first'].fillna(4)
#one_hot = pd.get_dummies(df['residence_type_first'])
#one_hot = one_hot.rename(columns={1.0: 'residence_first_with_DVV',2.0: 'residence_first_no_DVV',3.0: 'residence_first_foreign',4.0: 'residence_first_nan'})
#one_h = one_h.join(one_hot)
```


```python
df['mother_tongue']=df['mother_tongue'].fillna('unk')
one_hot = pd.get_dummies(df['mother_tongue'])
one_hot = one_hot.rename(columns={'fi': 'lang_fi','other': 'lang_other','ru': 'lang_ru','sv': 'lang_sv','unk': 'lang_unk'})
one_h = df[["FINREGISTRYID"]].join(one_hot)
```


```python
df['ever_married']=df['ever_married'].fillna(2)
one_hot = pd.get_dummies(df['ever_married'])
one_hot = one_hot.rename(columns={0.0: 'ever_married_no',1.0: 'ever_married_yes',2.0: 'ever_married_nan'})
one_h = one_h.join(one_hot)
```


```python
df['ever_divorced']=df['ever_divorced'].fillna(2)
one_hot = pd.get_dummies(df['ever_divorced'])
one_hot = one_hot.rename(columns={0.0: 'ever_divorced_no',1.0: 'ever_divorced_yes',2.0: 'ever_divorced_nan'})
one_h = one_h.join(one_hot)
```


```python
#df['emigrated']=df['emigrated'].fillna(2)
#one_hot = pd.get_dummies(df['emigrated'])
#one_hot = one_hot.rename(columns={0.0: 'emigrated_no',1.0: 'emigrated_yes',2.0: 'emigrated_nan'})
#one_h = one_h.join(one_hot)
```

# BIRTH


```python
birth = pd.read_csv('/data/processed_data/thl_birth/birth_2022-03-08.csv', sep=';',usecols=('AITI_TNRO','LAPSI_TNRO','TILASTOVUOSI','AITI_IKA','KESTOVKPV','SIVIILISAATY','AVOLIITTO','AIEMMATRASKAUDET','KESKENMENOJA',
                   'KESKEYTYKSIA','ULKOPUOLISIA','AIEMMATSYNNYTYKSET','KUOLLEENASYNT','TARKASTUKSET','POLILLA','TUPAKOINTITUNNUS',
                   'SYNNYTYSTAPATUNNUS','SIKIOITA','SYNTYMAPAINO','SYNTYMAPITUUS','ICD10_1','HOITOPAIKKATUNNUS',
                   'APGAR_1MIN','APGAR_5MIN','SOKERI_PATOL','ALKIONSIIRTO','PAINELUELVYTYS_ALKU','ELVYTYS_ALKU_JALKEEN'))
```


```python
z_scores = pd.read_csv('/data/projects/project_pvartiai/rsv/predictors/birth_size_sd_values.csv')
```


```python
birth = birth[birth['TILASTOVUOSI']<2018].copy() # remove data from 2018 in order for predictive interval info not to leak into traingin data
```


```python
birth = birth[birth['LAPSI_TNRO'].notna()].copy()
birth.rename(columns={'LAPSI_TNRO': "FINREGISTRYID"}, inplace=True)
```


```python
b_cont = birth[["FINREGISTRYID",'AITI_IKA','KESTOVKPV','SYNTYMAPAINO','SYNTYMAPITUUS']].copy() # mothers age / pregnancy duration / Birth weight / Birth length
b_cont.rename(columns={'AITI_IKA': "B_Mothers_age",'KESTOVKPV': "B_Pregnancy_duration",'SYNTYMAPAINO': "B_Birth_weight",'SYNTYMAPITUUS': "B_Birth length", }, inplace=True)
```


```python
b_cont['B_Pregnancy_duration']=b_cont['B_Pregnancy_duration'].apply(lambda x: int(x[:2])+int(x[3])/7 if(pd.notnull(x)) else x) # change duration format from "w+d" to w
```


```python
b_ord = birth[["FINREGISTRYID",'AIEMMATRASKAUDET','KESKENMENOJA','KESKEYTYKSIA','ULKOPUOLISIA','AIEMMATSYNNYTYKSET','KUOLLEENASYNT','TARKASTUKSET','POLILLA','SIKIOITA','APGAR_1MIN','APGAR_5MIN']].copy()
b_ord.rename(columns={'AIEMMATRASKAUDET': "B_prrevious_pregnancies",'KESKENMENOJA': "B_Previous_miscarriages",'KESKEYTYKSIA': "B_Previous_induced_abortions",'ULKOPUOLISIA': "B_Previous_ectopic_pregnancies",
                      'AIEMMATSYNNYTYKSET': "B_Previous_births",'KUOLLEENASYNT': "B_stilborn",'TARKASTUKSET': "B_check_ups",'POLILLA': "B_check_ups_outpat",'SIKIOITA': "B_Number_of_fetuses",'APGAR_1MIN': "B_apgar_1m",'APGAR_5MIN': "B_apgar_5m"}, inplace=True)
# prrevious pregnancies / Previous miscarriages / Previous induced abortions / Previous ectopic pregnancies / Previous births when at least one infant was stillborn / outpatient visits / Number of fetuses / Check-ups / Previous miscarriages / Apgar 1 / Apgar 7 /
```


```python
b_binary = birth[["FINREGISTRYID",'SOKERI_PATOL','ALKIONSIIRTO','PAINELUELVYTYS_ALKU','ELVYTYS_ALKU_JALKEEN']].copy() # Glucose tested and pathological / IVF, ICSI, FET /  Pressure resuscitation performed on the newborn / The child is resuscitated by the age of 7d
b_binary.rename(columns={'SOKERI_PATOL': "Glucose_pathological",'ALKIONSIIRTO': "B_IVF_ICSI_FET",'PAINELUELVYTYS_ALKU': "B_Pressure_resuscitation",'ELVYTYS_ALKU_JALKEEN': "B_resuscitated_by_7d"}, inplace=True)
```


```python
b_one_hot = birth[["FINREGISTRYID",'SIVIILISAATY','AVOLIITTO','TUPAKOINTITUNNUS','SYNNYTYSTAPATUNNUS', 'HOITOPAIKKATUNNUS']].copy() # Marital status, cohabiting, Maternal smoking, Mode of delivery, Child at 7 days
b_one_hot.rename(columns={'SIVIILISAATY': "B_Marital_status",'AVOLIITTO': "B_cohabiting",'TUPAKOINTITUNNUS': "B_Smoking",'SYNNYTYSTAPATUNNUS': "B_Delivery_mode",'HOITOPAIKKATUNNUS': "B_Child_7d"}, inplace=True)
```

### Replace NaN and merge to minimal phenotype


```python
continous = continous.merge(b_cont, on='FINREGISTRYID', how='left')

```


```python
# change variable because data in 2018-2019 is removed

print(binary['birth_registry_child'].value_counts())
binary['birth_registry_child']=0
binary.loc[continous['B_Mothers_age'].notna(), 'birth_registry_child'] = 1
print(binary['birth_registry_child'].value_counts())
```


```python
continous['B_Mothers_age'] = continous['B_Mothers_age'].fillna(continous['B_Mothers_age'].mean(skipna=True))
continous['B_Pregnancy_duration'] = continous['B_Pregnancy_duration'].fillna(continous['B_Pregnancy_duration'].mean(skipna=True))
continous['B_Birth_weight'] = continous['B_Birth_weight'].fillna(continous['B_Birth_weight'].mean(skipna=True))
continous['B_Birth length'] = continous['B_Birth length'].fillna(continous['B_Birth length'].mean(skipna=True))
```


```python
ordinal = ordinal.merge(b_ord, on='FINREGISTRYID', how='left')
```


```python
ordinal['B_prrevious_pregnancies'] = ordinal['B_prrevious_pregnancies'].fillna(ordinal['B_prrevious_pregnancies'].mode()[0])
ordinal['B_Previous_miscarriages'] = ordinal['B_Previous_miscarriages'].fillna(ordinal['B_Previous_miscarriages'].mode()[0])
ordinal['B_Previous_induced_abortions'] = ordinal['B_Previous_induced_abortions'].fillna(ordinal['B_Previous_induced_abortions'].mode()[0])
ordinal['B_Previous_ectopic_pregnancies'] = ordinal['B_Previous_ectopic_pregnancies'].fillna(ordinal['B_Previous_ectopic_pregnancies'].mode()[0])
ordinal['B_Previous_births'] = ordinal['B_Previous_births'].fillna(ordinal['B_Previous_births'].mode()[0])
ordinal['B_stilborn'] = ordinal['B_stilborn'].fillna(ordinal['B_stilborn'].mode()[0])
ordinal['B_check_ups'] = ordinal['B_check_ups'].fillna(ordinal['B_check_ups'].mode()[0])
ordinal['B_check_ups_outpat'] = ordinal['B_check_ups_outpat'].fillna(ordinal['B_check_ups_outpat'].mode()[0])
ordinal['B_Number_of_fetuses'] = ordinal['B_Number_of_fetuses'].fillna(ordinal['B_Number_of_fetuses'].mode()[0])
ordinal['B_apgar_1m'] = ordinal['B_apgar_1m'].fillna(ordinal['B_apgar_1m'].mode()[0])
ordinal['B_apgar_5m'] = ordinal['B_apgar_5m'].fillna(ordinal['B_apgar_5m'].mode()[0])
```


```python
binary = binary.merge(b_binary, on='FINREGISTRYID', how='left')
```


```python
binary['Glucose_pathological'] = binary['Glucose_pathological'].fillna(binary['Glucose_pathological'].mode()[0])
binary['B_IVF_ICSI_FET'] = binary['B_IVF_ICSI_FET'].fillna(binary['B_IVF_ICSI_FET'].mode()[0])
binary['B_Pressure_resuscitation'] = binary['B_Pressure_resuscitation'].fillna(binary['B_Pressure_resuscitation'].mode()[0])
binary['B_resuscitated_by_7d'] = binary['B_resuscitated_by_7d'].fillna(binary['B_resuscitated_by_7d'].mode()[0])
```


```python
one_h = one_h.merge(b_one_hot, on='FINREGISTRYID', how='left')
```


```python
one_h['B_Marital_status']=one_h['B_Marital_status'].fillna(one_h['B_Marital_status'].mode()[0])
one_hot = pd.get_dummies(one_h['B_Marital_status'])
one_hot = one_hot.rename(columns={0.0: 'B_Marital_status_0',1.0: 'B_Marital_status_1',2.0: 'B_Marital_status_2',3.0: 'B_Marital_status_3',4.0: 'B_Marital_status_4',5.0: 'B_Marital_status_5',6.0: 'B_Marital_status_6',7.0: 'B_Marital_status_7',8.0: 'B_Marital_status_8',9.0: 'B_Marital_status_9'})
one_h = one_h.join(one_hot)
del one_h['B_Marital_status']
```


```python
one_h['B_cohabiting']=one_h['B_cohabiting'].fillna(one_h['B_cohabiting'].mode()[0])
one_hot = pd.get_dummies(one_h['B_cohabiting'])
one_hot = one_hot.rename(columns={1.0: 'B_cohabiting_1',2.0: 'B_cohabiting_2',9.0: 'B_cohabiting_9'})
one_h = one_h.join(one_hot)
del one_h['B_cohabiting']
```


```python
one_h['B_Delivery_mode']=one_h['B_Delivery_mode'].fillna(one_h['B_Delivery_mode'].mode()[0])
one_hot = pd.get_dummies(one_h['B_Delivery_mode'])
one_hot = one_hot.rename(columns={1.0: 'B_Delivery_mode_1',2.0: 'B_Delivery_mode_2',3.0: 'B_Delivery_mode_3',4.0: 'B_Delivery_mode_4',5.0: 'B_Delivery_mode_5',6.0: 'B_Delivery_mode_6',7.0: 'B_Delivery_mode_7',8.0: 'B_Delivery_mode_8',9.0: 'B_Delivery_mode_9'})
one_h = one_h.join(one_hot)
del one_h['B_Delivery_mode']
```


```python
one_h['B_Child_7d']=one_h['B_Child_7d'].fillna(one_h['B_Child_7d'].mode()[0])
one_hot = pd.get_dummies(one_h['B_Child_7d'])
one_hot = one_hot.rename(columns={1.0: 'B_Child_7d_1',2.0: 'B_Child_7d_2',3.0: 'B_Child_7d_3',4.0: 'B_Child_7d_4',5.0: 'B_Child_7d_5',9.0: 'B_Child_7d_9'})
one_h = one_h.join(one_hot)
del one_h['B_Child_7d']
```


```python
one_h['B_Smoking']=one_h['B_Smoking'].fillna(one_h['B_Smoking'].mode()[0])
one_hot = pd.get_dummies(one_h['B_Smoking'])
one_hot = one_hot.rename(columns={1.0: 'B_Smoking_1',2.0: 'B_Smoking_2',3.0: 'B_Smoking_3',4.0: 'B_Smoking_4',9.0: 'B_Smoking_9'})
one_h = one_h.join(one_hot)
del one_h['B_Smoking']
```


```python
print(continous.shape,ordinal.shape,binary.shape,one_h.shape)
```

# MALFORMATIONS


```python
malform = pd.read_csv('/data/processed_data/thl_malformations/malformations_basic_2022-01-26.csv', sep=';')
```


```python
malform = malform[malform['YEAR_OF_BIRTH']<2018].copy() # remove data from 2018 in order for predictive interval info not to leak into traingin data
```


```python
malform.rename(columns={'TNRO': "FINREGISTRYID"}, inplace=True)
one_h = one_h.merge(malform[['FINREGISTRYID','PATTERN']], on='FINREGISTRYID', how='left')
one_h['PATTERN']=one_h['PATTERN'].fillna(9.0)
```


```python
one_hot = pd.get_dummies(one_h['PATTERN'])
one_hot = one_hot.rename(columns={0.0: 'M_PATTERN_0',1.0: 'M_PATTERN_1',2.0: 'M_PATTERN_2',3.0: 'M_PATTERN_3',4.0: 'M_PATTERN_4',8.0: 'M_PATTERN_8',9.0: 'M_PATTERN_9'})
one_h = one_h.join(one_hot)
del one_h['PATTERN']
```


```python
# tgwew is an accurate binary cariable for who is in malformation reg 'M_PATTERN_9'
del binary['in_malformations_registry']
```

# SOCIAL ASSITANCE


```python
ass_spouse = pd.read_csv('/data/processed_data/thl_social_assistance/3214_FinRegistry_puolisontoitu_MattssonHannele07122020.csv.finreg_IDsp',sep=';')
```


```python
ass = pd.read_csv('/data/processed_data/thl_social_assistance/3214_FinRegistry_toitu_MattssonHannele07122020.csv.finreg_IDsp',sep=';')
```


```python
del ass_spouse['HAKIJA']
```


```python
ass_all = ass.append(ass_spouse, ignore_index=True)
```


```python
ass_all['TNRO'].nunique()
```


```python
ass_all = ass_all[ass_all['TILASTOVUOSI']<2018].copy() # remove data from 2018 in order for predictive interval info not to leak into traingin data
```


```python
lines = []
total = ass_all['TNRO'].nunique()
grouped=ass_all.groupby(ass_all['TNRO'])
for g,f in tqdm(grouped, total=total):
    benefit = f['VARS_TOIMEENTULOTUKI_EUR'].sum()
    benefit_months = f['VARS_TOIMEENTULOTUKI_KK'].sum()
    benefoit_years = f.shape[0]
    line = [g,benefit,benefit_months,benefoit_years] #' '.join(sources),
    lines.append(line)
grouped = pd.DataFrame(lines,columns = ["FINREGISTRYID","Total_assistance","Assistance_months", "Assistance_years"])
#for g, f in ass_all.groupby(ass_all['TNRO'])  
```


```python
ass_all['TNRO'].nunique()
```


```python
grouped["Total_assistance"].mean()/grouped["Assistance_months"].mean()
```


```python
continous = continous.merge(grouped, on='FINREGISTRYID', how='left')
```


```python
continous["Total_assistance"]=continous["Total_assistance"].fillna(0)
continous["Assistance_months"]=continous["Assistance_months"].fillna(0)
continous["Assistance_years"]=continous["Assistance_years"].fillna(0)
```


```python
# change variable because data in 2018-2019 is removed

print(binary['in_social_assistance_registries'].value_counts())
binary['in_social_assistance_registries']=0
binary.loc[continous["Total_assistance"]!=0, 'in_social_assistance_registries'] = 1
print(binary['in_social_assistance_registries'].value_counts())
```

# SOCIAL HILMO


```python
soc_h = pd.read_csv('/data/processed_data/thl_soshilmo/thl2019_1776_soshilmo.csv.finreg_IDsp')
```


```python
soc_h = soc_h[soc_h['VUOSI']<2018].copy() # remove data from 2018 in order for predictive interval info not to leak into traingin data
```


```python
#soc_h[soc_h['TNRO']=='FR2697827'][['VUOSI','TUPVA','LPVM','HOITOPV','KVHP']]
```


```python
lines = []
total = soc_h['TNRO'].nunique()
grouped=soc_h.groupby(soc_h['TNRO'])
for g,f in tqdm(grouped, total=total):
    duration_days = f['KVHP'].sum()
    line = [g,duration_days] #' '.join(sources),
    lines.append(line)
grouped = pd.DataFrame(lines,columns = ["FINREGISTRYID","SocHilm_Duration_days"])
```


```python
grouped.loc[grouped["SocHilm_Duration_days"]<0, "SocHilm_Duration_days"] = 1 #replace negatives with 1
```


```python
continous = continous.merge(grouped, on='FINREGISTRYID', how='left')
```


```python
continous["SocHilm_Duration_days"]=continous["SocHilm_Duration_days"].fillna(0)
```


```python
# change variable because data in 2018-2019 is removed

print(binary['in_social_hilmo'].value_counts())
binary['in_social_hilmo']=0
binary.loc[continous["SocHilm_Duration_days"]!=0, 'in_social_hilmo'] = 1
print(binary['in_social_hilmo'].value_counts())
```

# Intensive care


```python
Intense = pd.read_csv('/data/processed_data/ficc_intensive_care/intensive_care_2022-01-20.csv', sep = ";")
```


```python
Intense = Intense[Intense['ADM_TIME'].notna()].copy()
```


```python
Intense['ADM_TIME'] =Intense['ADM_TIME'].apply(lambda x: x[:4])
```


```python
Intense['ADM_TIME'] = Intense['ADM_TIME'].astype(int)
```


```python
Intense = Intense[Intense['ADM_TIME']<2018].copy() # remove data from 2018 in order for predictive interval info not to leak into traingin data
```


```python
Intense['ADM_TIME'].value_counts()
```


```python
# There is nothing to include as intensi ve cre reister starts in 2020
```

# Smoking status


```python
path = '/data/processed_data/thl_avohilmo/thl2019_1776_avohilmo_11_12.csv.finreg_IDsp'
start_time = time.time()
a11_12 = pd.read_csv(path,usecols=['TNRO','KAYNTI_ALKOI','TUPAKOINTI'])
a11_12 = a11_12[a11_12['TUPAKOINTI'].notna()].copy()
run_time = time.time()-start_time
print(run_time)

path = '/data/processed_data/thl_avohilmo/thl2019_1776_avohilmo_13_14.csv.finreg_IDsp'
start_time = time.time()
a13_14 = pd.read_csv(path,usecols=['TNRO','KAYNTI_ALKOI','TUPAKOINTI'])
a13_14 = a13_14[a13_14['TUPAKOINTI'].notna()].copy()
run_time = time.time()-start_time
print(run_time)


path = '/data/processed_data/thl_avohilmo/thl2019_1776_avohilmo_15_16.csv.finreg_IDsp'
start_time = time.time()
a15_16 = pd.read_csv(path,usecols=['TNRO','KAYNTI_ALKOI','TUPAKOINTI'])
a15_16 = a15_16[a15_16['TUPAKOINTI'].notna()].copy()
run_time = time.time()-start_time
print(run_time)
```


```python
path = '/data/processed_data/thl_avohilmo/thl2019_1776_avohilmo_17_18.csv.finreg_IDsp'
start_time = time.time()
a17_18 = pd.read_csv(path,usecols=['TNRO','KAYNTI_ALKOI','TUPAKOINTI'])
a17_18 = a17_18[a17_18['TUPAKOINTI'].notna()].copy()
run_time = time.time()-start_time
print(run_time)

path = '/data/processed_data/thl_avohilmo/thl2019_1776_avohilmo_19_20.csv.finreg_IDsp'
start_time = time.time()
a19_20 = pd.read_csv(path,usecols=['TNRO','KAYNTI_ALKOI','TUPAKOINTI'])
a19_20 = a19_20[a19_20['TUPAKOINTI'].notna()].copy()
run_time = time.time()-start_time
print(run_time)
```


```python
print(a11_12.shape,a13_14.shape,a15_16.shape,a17_18.shape,a19_20.shape)
```


```python
print(a11_12['TUPAKOINTI'].value_counts(),a13_14['TUPAKOINTI'].value_counts(),a15_16['TUPAKOINTI'].value_counts(),a17_18['TUPAKOINTI'].value_counts(),a19_20['TUPAKOINTI'].value_counts())
```


```python
# Data cleaning
a11_12 = a11_12[a11_12['TUPAKOINTI']!=-2].copy()

a13_14 = a13_14[a13_14['TUPAKOINTI']!=-2].copy()
a13_14 = a13_14[a13_14['TUPAKOINTI']!=0].copy()
a13_14 = a13_14[a13_14['TUPAKOINTI']!=5].copy()

a15_16 = a15_16[a15_16['TUPAKOINTI']!=0].copy()
a15_16 = a15_16[a15_16['TUPAKOINTI']!=5].copy()
a15_16 = a15_16[a15_16['TUPAKOINTI']!=-2].copy()

a17_18 = a17_18[a17_18['TUPAKOINTI']!=0].copy()
a17_18 = a17_18[a17_18['TUPAKOINTI']!=5].copy()
a17_18 = a17_18[a17_18['TUPAKOINTI']!=8].copy()
a17_18 = a17_18[a17_18['TUPAKOINTI']!=-2].copy()

a19_20 = a19_20[a19_20['TUPAKOINTI']!=0].copy()
a19_20 = a19_20[a19_20['TUPAKOINTI']!=-1].copy()
a19_20 = a19_20[a19_20['TUPAKOINTI']!=5].copy()
a19_20 = a19_20[a19_20['TUPAKOINTI']!=-2].copy()
```


```python
smok = pd.concat([a11_12, a13_14, a15_16, a17_18, a19_20], ignore_index=True)
```


```python
smok['TUPAKOINTI'].value_counts()
```


```python
print((a11_12.shape[0]+a13_14.shape[0]+a15_16.shape[0]+a17_18.shape[0]+a19_20.shape[0])==smok.shape[0])
```


```python
smok['TNRO'].nunique()
```


```python
# Add smoking status from birth register
```


```python
b_smok = pd.read_csv('/data/processed_data/thl_birth/birth_2022-03-08.csv', sep=';',usecols=('AITI_TNRO','LAPSEN_SYNTYMAPVM','TUPAKOINTITUNNUS'))
b_smok = b_smok.rename(columns={'AITI_TNRO': 'TNRO', 'LAPSEN_SYNTYMAPVM': 'KAYNTI_ALKOI', 'TUPAKOINTITUNNUS': 'TUPAKOINTI'})
b_smok = b_smok[['TNRO','KAYNTI_ALKOI','TUPAKOINTI']]
smoke = pd.concat([smok, b_smok], ignore_index=True)
print((smok.shape[0]+b_smok.shape[0])==smoke.shape[0])
```


```python
# leave only last recorded smoke status for each ID
smoke = smoke.rename(columns={'TNRO': "FINREGISTRYID",'KAYNTI_ALKOI': "PVM", 'TUPAKOINTI': "Smk_status"})
smoke['year']=smoke['PVM'].apply(lambda x: x[:4])
smoke = smoke.sort_values(["FINREGISTRYID", "year"], ascending = (True, False))
duplicates = smoke.duplicated(subset=["FINREGISTRYID"])
smoke = smoke[duplicates == False].copy()
```


```python
smoke["FINREGISTRYID"].nunique()
```


```python
one_h = one_h.merge(smoke[['FINREGISTRYID',"Smk_status"]], on='FINREGISTRYID', how='left')
one_h["Smk_status"]=one_h["Smk_status"].fillna(8.0) # IDs for whom there is no smoking status recorded are relaced with 8
```


```python
one_hot = pd.get_dummies(one_h["Smk_status"])
one_hot = one_hot.rename(columns={1.0: 'Smk_status_1',2.0: 'Smk_status_2',3.0: 'Smk_status_3',4.0: 'Smk_status_4',8.0: 'Smk_status_8',9.0: 'Smk_status_9'})
one_h = one_h.join(one_hot)
del one_h["Smk_status"]
```


```python
# combine different types of features together
features = continous.copy()
features = features.merge(ordinal, on='FINREGISTRYID', how='left')
features = features.merge(binary, on='FINREGISTRYID', how='left')
features = features.merge(one_h, on='FINREGISTRYID', how='left')
```


```python
features = features.sort_values(["FINREGISTRYID"], ascending = (True))
features.reset_index(drop=True, inplace=True)
```


```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# transform continous features
features[['AGE','B_Mothers_age', 'B_Pregnancy_duration', 'B_Birth_weight','B_Birth length', 'Total_assistance',
          'Assistance_months','Assistance_years', 'SocHilm_Duration_days']] = scaler.fit_transform(features[['AGE',
          'B_Mothers_age', 'B_Pregnancy_duration', 'B_Birth_weight','B_Birth length', 'Total_assistance', 'Assistance_months','Assistance_years', 'SocHilm_Duration_days']])
# transform ordinal features
features[['number_of_children', 'drug_purchases', 'kanta_prescriptions','B_prrevious_pregnancies', 'B_Previous_miscarriages', 'B_Previous_induced_abortions','B_Previous_ectopic_pregnancies', 'B_Previous_births', 'B_stilborn','B_check_ups', 'B_check_ups_outpat',
          'B_Number_of_fetuses','B_apgar_1m', 'B_apgar_5m']] = scaler.fit_transform(features[['number_of_children', 'drug_purchases', 'kanta_prescriptions','B_prrevious_pregnancies', 'B_Previous_miscarriages', 'B_Previous_induced_abortions','B_Previous_ectopic_pregnancies',
          'B_Previous_births', 'B_stilborn','B_check_ups', 'B_check_ups_outpat','B_Number_of_fetuses','B_apgar_1m', 'B_apgar_5m']])
```


```python
features.to_csv('/data/projects/project_avabalas/RNN/preprocessing_new/demographic_features.csv', index=False)
```


```python
# " ".join demograophic features to reduce computational reecoourse requirements down the line
demo = pd.read_csv('/data/projects/project_avabalas/RNN/preprocessing_new/demographic_features.csv', dtype=str)
demo = demo[demo['sex'].notna()]
demo = pd.concat([demo[['FINREGISTRYID']], demo[demo.iloc[:,1:].columns.tolist()].T.agg(' '.join)], axis = 1)
demo = demo.rename(columns={'0': 'demo'})
demo.to_csv('/data/projects/project_avabalas/RNN/preprocessing_new/demographic_features2.csv', index=False)
```
