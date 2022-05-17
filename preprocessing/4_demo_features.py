#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# # Minimal phenotype

# In[ ]:


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


# In[ ]:


continous = df[["FINREGISTRYID",'AGE']].copy()


# In[ ]:


ordinal = df[["FINREGISTRYID",'number_of_children','drug_purchases','kanta_prescriptions']].copy()


# In[ ]:


ordinal['number_of_children'] = ordinal['number_of_children'].fillna(ordinal['number_of_children'].mean(skipna=True))
ordinal['drug_purchases'] = ordinal['drug_purchases'].fillna(ordinal['drug_purchases'].mean(skipna=True))
ordinal['kanta_prescriptions'] = ordinal['kanta_prescriptions'].fillna(ordinal['kanta_prescriptions'].mean(skipna=True))


# In[ ]:


binary = df[["FINREGISTRYID",'sex','in_social_assistance_registries','in_social_hilmo','index_person','ever_married','ever_divorced','emigrated',
         'birth_registry_mother','birth_registry_child','in_vaccination_registry','in_infect_dis_registry','in_malformations_registry','in_cancer_registry']].copy()


# In[ ]:


#df['residence_type_last']=df['residence_type_last'].fillna(4)
#one_hot = pd.get_dummies(df['residence_type_last'])
#one_hot = one_hot.rename(columns={1.0: 'residence_with_DVV',2.0: 'residence_no_DVV',3.0: 'residence_foreign',4.0: 'residence_nan'})
#one_h = df[["FINREGISTRYID"]].join(one_hot.copy())

#df['residence_type_first']=df['residence_type_first'].fillna(4)
#one_hot = pd.get_dummies(df['residence_type_first'])
#one_hot = one_hot.rename(columns={1.0: 'residence_first_with_DVV',2.0: 'residence_first_no_DVV',3.0: 'residence_first_foreign',4.0: 'residence_first_nan'})
#one_h = one_h.join(one_hot)


# In[ ]:


df['mother_tongue']=df['mother_tongue'].fillna('unk')
one_hot = pd.get_dummies(df['mother_tongue'])
one_hot = one_hot.rename(columns={'fi': 'lang_fi','other': 'lang_other','ru': 'lang_ru','sv': 'lang_sv','unk': 'lang_unk'})
one_h = df[["FINREGISTRYID"]].join(one_hot)


# # BIRTH

# In[ ]:


birth = pd.read_csv('/data/processed_data/thl_birth/birth_2022-03-08.csv', sep=';',usecols=('AITI_TNRO','LAPSI_TNRO','TILASTOVUOSI','AITI_IKA','KESTOVKPV','SIVIILISAATY','AVOLIITTO','AIEMMATRASKAUDET','KESKENMENOJA',
                   'KESKEYTYKSIA','ULKOPUOLISIA','AIEMMATSYNNYTYKSET','KUOLLEENASYNT','TARKASTUKSET','POLILLA','TUPAKOINTITUNNUS',
                   'SYNNYTYSTAPATUNNUS','SIKIOITA','SYNTYMAPAINO','SYNTYMAPITUUS','ICD10_1','HOITOPAIKKATUNNUS',
                   'APGAR_1MIN','APGAR_5MIN','SOKERI_PATOL','ALKIONSIIRTO','PAINELUELVYTYS_ALKU','ELVYTYS_ALKU_JALKEEN'))


# In[ ]:


z_scores = pd.read_csv('/data/projects/project_pvartiai/rsv/predictors/birth_size_sd_values.csv')


# In[ ]:


birth = birth[birth['TILASTOVUOSI']<2018].copy() # remove data from 2018 in order for predictive interval info not to leak into traingin data


# In[ ]:


birth = birth[birth['LAPSI_TNRO'].notna()].copy()
birth.rename(columns={'LAPSI_TNRO': "FINREGISTRYID"}, inplace=True)


# In[ ]:


b_cont = birth[["FINREGISTRYID",'AITI_IKA','KESTOVKPV','SYNTYMAPAINO','SYNTYMAPITUUS']].copy() # mothers age / pregnancy duration / Birth weight / Birth length
b_cont.rename(columns={'AITI_IKA': "B_Mothers_age",'KESTOVKPV': "B_Pregnancy_duration",'SYNTYMAPAINO': "B_Birth_weight",'SYNTYMAPITUUS': "B_Birth length", }, inplace=True)


# In[ ]:


b_cont['B_Pregnancy_duration']=b_cont['B_Pregnancy_duration'].apply(lambda x: int(x[:2])+int(x[3])/7 if(pd.notnull(x)) else x) # change duration format from "w+d" to w


# In[ ]:


b_ord = birth[["FINREGISTRYID",'AIEMMATRASKAUDET','KESKENMENOJA','KESKEYTYKSIA','ULKOPUOLISIA','AIEMMATSYNNYTYKSET','KUOLLEENASYNT','TARKASTUKSET','POLILLA','SIKIOITA','APGAR_1MIN','APGAR_5MIN']].copy()
b_ord.rename(columns={'AIEMMATRASKAUDET': "B_prrevious_pregnancies",'KESKENMENOJA': "B_Previous_miscarriages",'KESKEYTYKSIA': "B_Previous_induced_abortions",'ULKOPUOLISIA': "B_Previous_ectopic_pregnancies",
                      'AIEMMATSYNNYTYKSET': "B_Previous_births",'KUOLLEENASYNT': "B_stilborn",'TARKASTUKSET': "B_check_ups",'POLILLA': "B_check_ups_outpat",'SIKIOITA': "B_Number_of_fetuses",'APGAR_1MIN': "B_apgar_1m",'APGAR_5MIN': "B_apgar_5m"}, inplace=True)
# prrevious pregnancies / Previous miscarriages / Previous induced abortions / Previous ectopic pregnancies / Previous births when at least one infant was stillborn / outpatient visits / Number of fetuses / Check-ups / Previous miscarriages / Apgar 1 / Apgar 7 /


# In[ ]:


b_binary = birth[["FINREGISTRYID",'SOKERI_PATOL','ALKIONSIIRTO','PAINELUELVYTYS_ALKU','ELVYTYS_ALKU_JALKEEN']].copy() # Glucose tested and pathological / IVF, ICSI, FET /  Pressure resuscitation performed on the newborn / The child is resuscitated by the age of 7d
b_binary.rename(columns={'SOKERI_PATOL': "Glucose_pathological",'ALKIONSIIRTO': "B_IVF_ICSI_FET",'PAINELUELVYTYS_ALKU': "B_Pressure_resuscitation",'ELVYTYS_ALKU_JALKEEN': "B_resuscitated_by_7d"}, inplace=True)


# In[ ]:


b_one_hot = birth[["FINREGISTRYID",'SIVIILISAATY','AVOLIITTO','TUPAKOINTITUNNUS','SYNNYTYSTAPATUNNUS', 'HOITOPAIKKATUNNUS']].copy() # Marital status, cohabiting, Maternal smoking, Mode of delivery, Child at 7 days
b_one_hot.rename(columns={'SIVIILISAATY': "B_Marital_status",'AVOLIITTO': "B_cohabiting",'TUPAKOINTITUNNUS': "B_Smoking",'SYNNYTYSTAPATUNNUS': "B_Delivery_mode",'HOITOPAIKKATUNNUS': "B_Child_7d"}, inplace=True)


# ### Replace NaN and merge to minimal phenotype

# In[ ]:


continous = continous.merge(b_cont, on='FINREGISTRYID', how='left')


# In[ ]:


# change variable because data in 2018-2019 is removed

print(binary['birth_registry_child'].value_counts())
binary['birth_registry_child']=0
binary.loc[continous['B_Mothers_age'].notna(), 'birth_registry_child'] = 1
print(binary['birth_registry_child'].value_counts())


# In[ ]:


continous['B_Mothers_age'] = continous['B_Mothers_age'].fillna(continous['B_Mothers_age'].mean(skipna=True))
continous['B_Pregnancy_duration'] = continous['B_Pregnancy_duration'].fillna(continous['B_Pregnancy_duration'].mean(skipna=True))
continous['B_Birth_weight'] = continous['B_Birth_weight'].fillna(continous['B_Birth_weight'].mean(skipna=True))
continous['B_Birth length'] = continous['B_Birth length'].fillna(continous['B_Birth length'].mean(skipna=True))


# In[ ]:


ordinal = ordinal.merge(b_ord, on='FINREGISTRYID', how='left')


# In[ ]:


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


# In[ ]:


binary = binary.merge(b_binary, on='FINREGISTRYID', how='left')


# In[ ]:


binary['Glucose_pathological'] = binary['Glucose_pathological'].fillna(binary['Glucose_pathological'].mode()[0])
binary['B_IVF_ICSI_FET'] = binary['B_IVF_ICSI_FET'].fillna(binary['B_IVF_ICSI_FET'].mode()[0])
binary['B_Pressure_resuscitation'] = binary['B_Pressure_resuscitation'].fillna(binary['B_Pressure_resuscitation'].mode()[0])
binary['B_resuscitated_by_7d'] = binary['B_resuscitated_by_7d'].fillna(binary['B_resuscitated_by_7d'].mode()[0])


# In[ ]:


one_h = one_h.merge(b_one_hot, on='FINREGISTRYID', how='left')


# In[ ]:


one_h['B_Marital_status']=one_h['B_Marital_status'].fillna(one_h['B_Marital_status'].mode()[0])
one_hot = pd.get_dummies(one_h['B_Marital_status'])
one_hot = one_hot.rename(columns={0.0: 'B_Marital_status_0',1.0: 'B_Marital_status_1',2.0: 'B_Marital_status_2',3.0: 'B_Marital_status_3',4.0: 'B_Marital_status_4',5.0: 'B_Marital_status_5',6.0: 'B_Marital_status_6',7.0: 'B_Marital_status_7',8.0: 'B_Marital_status_8',9.0: 'B_Marital_status_9'})
one_h = one_h.join(one_hot)
del one_h['B_Marital_status']


# In[ ]:


one_h['B_cohabiting']=one_h['B_cohabiting'].fillna(one_h['B_cohabiting'].mode()[0])
one_hot = pd.get_dummies(one_h['B_cohabiting'])
one_hot = one_hot.rename(columns={1.0: 'B_cohabiting_1',2.0: 'B_cohabiting_2',9.0: 'B_cohabiting_9'})
one_h = one_h.join(one_hot)
del one_h['B_cohabiting']


# In[ ]:


one_h['B_Delivery_mode']=one_h['B_Delivery_mode'].fillna(one_h['B_Delivery_mode'].mode()[0])
one_hot = pd.get_dummies(one_h['B_Delivery_mode'])
one_hot = one_hot.rename(columns={1.0: 'B_Delivery_mode_1',2.0: 'B_Delivery_mode_2',3.0: 'B_Delivery_mode_3',4.0: 'B_Delivery_mode_4',5.0: 'B_Delivery_mode_5',6.0: 'B_Delivery_mode_6',7.0: 'B_Delivery_mode_7',8.0: 'B_Delivery_mode_8',9.0: 'B_Delivery_mode_9'})
one_h = one_h.join(one_hot)
del one_h['B_Delivery_mode']


# In[ ]:


one_h['B_Child_7d']=one_h['B_Child_7d'].fillna(one_h['B_Child_7d'].mode()[0])
one_hot = pd.get_dummies(one_h['B_Child_7d'])
one_hot = one_hot.rename(columns={1.0: 'B_Child_7d_1',2.0: 'B_Child_7d_2',3.0: 'B_Child_7d_3',4.0: 'B_Child_7d_4',5.0: 'B_Child_7d_5',9.0: 'B_Child_7d_9'})
one_h = one_h.join(one_hot)
del one_h['B_Child_7d']


# In[ ]:


one_h['B_Smoking']=one_h['B_Smoking'].fillna(one_h['B_Smoking'].mode()[0])
one_hot = pd.get_dummies(one_h['B_Smoking'])
one_hot = one_hot.rename(columns={1.0: 'B_Smoking_1',2.0: 'B_Smoking_2',3.0: 'B_Smoking_3',4.0: 'B_Smoking_4',9.0: 'B_Smoking_9'})
one_h = one_h.join(one_hot)
del one_h['B_Smoking']


# In[ ]:


print(continous.shape,ordinal.shape,binary.shape,one_h.shape)


# # MALFORMATIONS

# In[ ]:


malform = pd.read_csv('/data/processed_data/thl_malformations/malformations_basic_2022-01-26.csv', sep=';')


# In[ ]:


malform = malform[malform['YEAR_OF_BIRTH']<2018].copy() # remove data from 2018 in order for predictive interval info not to leak into traingin data


# In[ ]:


malform.rename(columns={'TNRO': "FINREGISTRYID"}, inplace=True)
one_h = one_h.merge(malform[['FINREGISTRYID','PATTERN']], on='FINREGISTRYID', how='left')
one_h['PATTERN']=one_h['PATTERN'].fillna(9.0)


# In[ ]:


one_hot = pd.get_dummies(one_h['PATTERN'])
one_hot = one_hot.rename(columns={0.0: 'M_PATTERN_0',1.0: 'M_PATTERN_1',2.0: 'M_PATTERN_2',3.0: 'M_PATTERN_3',4.0: 'M_PATTERN_4',8.0: 'M_PATTERN_8',9.0: 'M_PATTERN_9'})
one_h = one_h.join(one_hot)
del one_h['PATTERN']


# In[ ]:


# tgwew is an accurate binary cariable for who is in malformation reg 'M_PATTERN_9'
del binary['in_malformations_registry']


# # SOCIAL ASSITANCE

# In[ ]:


ass_spouse = pd.read_csv('/data/processed_data/thl_social_assistance/3214_FinRegistry_puolisontoitu_MattssonHannele07122020.csv.finreg_IDsp',sep=';')


# In[ ]:


ass = pd.read_csv('/data/processed_data/thl_social_assistance/3214_FinRegistry_toitu_MattssonHannele07122020.csv.finreg_IDsp',sep=';')


# In[ ]:


del ass_spouse['HAKIJA']


# In[ ]:


ass_all = ass.append(ass_spouse, ignore_index=True)


# In[ ]:


ass_all['TNRO'].nunique()


# In[ ]:


ass_all = ass_all[ass_all['TILASTOVUOSI']<2018].copy() # remove data from 2018 in order for predictive interval info not to leak into traingin data


# In[ ]:


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


# In[ ]:


ass_all['TNRO'].nunique()


# In[ ]:


grouped["Total_assistance"].mean()/grouped["Assistance_months"].mean()


# In[ ]:


continous = continous.merge(grouped, on='FINREGISTRYID', how='left')


# In[ ]:


continous["Total_assistance"]=continous["Total_assistance"].fillna(0)
continous["Assistance_months"]=continous["Assistance_months"].fillna(0)
continous["Assistance_years"]=continous["Assistance_years"].fillna(0)


# In[ ]:


# change variable because data in 2018-2019 is removed

print(binary['in_social_assistance_registries'].value_counts())
binary['in_social_assistance_registries']=0
binary.loc[continous["Total_assistance"]!=0, 'in_social_assistance_registries'] = 1
print(binary['in_social_assistance_registries'].value_counts())


# # SOCIAL HILMO

# In[ ]:


soc_h = pd.read_csv('/data/processed_data/thl_soshilmo/thl2019_1776_soshilmo.csv.finreg_IDsp')


# In[ ]:


soc_h = soc_h[soc_h['VUOSI']<2018].copy() # remove data from 2018 in order for predictive interval info not to leak into traingin data


# In[ ]:


#soc_h[soc_h['TNRO']=='FR2697827'][['VUOSI','TUPVA','LPVM','HOITOPV','KVHP']]


# In[ ]:


lines = []
total = soc_h['TNRO'].nunique()
grouped=soc_h.groupby(soc_h['TNRO'])
for g,f in tqdm(grouped, total=total):
    duration_days = f['KVHP'].sum()
    line = [g,duration_days] #' '.join(sources),
    lines.append(line)
grouped = pd.DataFrame(lines,columns = ["FINREGISTRYID","SocHilm_Duration_days"])


# In[ ]:


grouped.loc[grouped["SocHilm_Duration_days"]<0, "SocHilm_Duration_days"] = 1 #replace negatives with 1


# In[ ]:


continous = continous.merge(grouped, on='FINREGISTRYID', how='left')


# In[ ]:


continous["SocHilm_Duration_days"]=continous["SocHilm_Duration_days"].fillna(0)


# In[ ]:


# change variable because data in 2018-2019 is removed

print(binary['in_social_hilmo'].value_counts())
binary['in_social_hilmo']=0
binary.loc[continous["SocHilm_Duration_days"]!=0, 'in_social_hilmo'] = 1
print(binary['in_social_hilmo'].value_counts())


# # Intensive care

# In[ ]:


Intense = pd.read_csv('/data/processed_data/ficc_intensive_care/intensive_care_2022-01-20.csv', sep = ";")


# In[ ]:


Intense = Intense[Intense['ADM_TIME'].notna()].copy()


# In[ ]:


Intense['ADM_TIME'] =Intense['ADM_TIME'].apply(lambda x: x[:4])


# In[ ]:


Intense['ADM_TIME'] = Intense['ADM_TIME'].astype(int)


# In[ ]:


Intense = Intense[Intense['ADM_TIME']<2018].copy() # remove data from 2018 in order for predictive interval info not to leak into traingin data


# In[ ]:


Intense['ADM_TIME'].value_counts()


# In[ ]:


# There is nothing to include as intensi ve cre reister starts in 2020


# # Smoking status

# In[ ]:


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


# In[ ]:


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


# In[ ]:


print(a11_12.shape,a13_14.shape,a15_16.shape,a17_18.shape,a19_20.shape)


# In[ ]:


print(a11_12['TUPAKOINTI'].value_counts(),a13_14['TUPAKOINTI'].value_counts(),a15_16['TUPAKOINTI'].value_counts(),a17_18['TUPAKOINTI'].value_counts(),a19_20['TUPAKOINTI'].value_counts())


# In[ ]:


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


# In[ ]:


smok = pd.concat([a11_12, a13_14, a15_16, a17_18, a19_20], ignore_index=True)


# In[ ]:


smok['TUPAKOINTI'].value_counts()


# In[ ]:


print((a11_12.shape[0]+a13_14.shape[0]+a15_16.shape[0]+a17_18.shape[0]+a19_20.shape[0])==smok.shape[0])


# In[ ]:


smok['TNRO'].nunique()


# In[ ]:


# Add smoking status from birth register


# In[ ]:


b_smok = pd.read_csv('/data/processed_data/thl_birth/birth_2022-03-08.csv', sep=';',usecols=('AITI_TNRO','LAPSEN_SYNTYMAPVM','TUPAKOINTITUNNUS'))
b_smok = b_smok.rename(columns={'AITI_TNRO': 'TNRO', 'LAPSEN_SYNTYMAPVM': 'KAYNTI_ALKOI', 'TUPAKOINTITUNNUS': 'TUPAKOINTI'})
b_smok = b_smok[['TNRO','KAYNTI_ALKOI','TUPAKOINTI']]
smoke = pd.concat([smok, b_smok], ignore_index=True)
print((smok.shape[0]+b_smok.shape[0])==smoke.shape[0])


# In[ ]:


# leave only last recorded smoke status for each ID
smoke = smoke.rename(columns={'TNRO': "FINREGISTRYID",'KAYNTI_ALKOI': "PVM", 'TUPAKOINTI': "Smk_status"})
smoke['year']=smoke['PVM'].apply(lambda x: x[:4])
smoke = smoke.sort_values(["FINREGISTRYID", "year"], ascending = (True, False))
duplicates = smoke.duplicated(subset=["FINREGISTRYID"])
smoke = smoke[duplicates == False].copy()


# In[ ]:


smoke["FINREGISTRYID"].nunique()


# In[ ]:


one_h = one_h.merge(smoke[['FINREGISTRYID',"Smk_status"]], on='FINREGISTRYID', how='left')
one_h["Smk_status"]=one_h["Smk_status"].fillna(8.0) # IDs for whom there is no smoking status recorded are relaced with 8


# In[ ]:


one_hot = pd.get_dummies(one_h["Smk_status"])
one_hot = one_hot.rename(columns={1.0: 'Smk_status_1',2.0: 'Smk_status_2',3.0: 'Smk_status_3',4.0: 'Smk_status_4',8.0: 'Smk_status_8',9.0: 'Smk_status_9'})
one_h = one_h.join(one_hot)
del one_h["Smk_status"]


# In[ ]:


# combine different types of features together
features = continous.copy()
features = features.merge(ordinal, on='FINREGISTRYID', how='left')
features = features.merge(binary, on='FINREGISTRYID', how='left')
features = features.merge(one_h, on='FINREGISTRYID', how='left')


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# transform continous features
features[['AGE', 'longitude_last', 'latitude_last','latitude_first', 'longitude_first', 'residence_duration_first','B_Mothers_age', 'B_Pregnancy_duration', 'B_Birth_weight','B_Birth length', 'Total_assistance',
          'Assistance_months','Assistance_years', 'SocHilm_Duration_days']] = scaler.fit_transform(features[['AGE', 'longitude_last', 'latitude_last','latitude_first', 'longitude_first', 'residence_duration_first',
          'B_Mothers_age', 'B_Pregnancy_duration', 'B_Birth_weight','B_Birth length', 'Total_assistance', 'Assistance_months','Assistance_years', 'SocHilm_Duration_days']])
# transform ordinal features
features[['number_of_children', 'drug_purchases', 'kanta_prescriptions','B_prrevious_pregnancies', 'B_Previous_miscarriages', 'B_Previous_induced_abortions','B_Previous_ectopic_pregnancies', 'B_Previous_births', 'B_stilborn','B_check_ups', 'B_check_ups_outpat',
          'B_Number_of_fetuses','B_apgar_1m', 'B_apgar_5m']] = scaler.fit_transform(features[['number_of_children', 'drug_purchases', 'kanta_prescriptions','B_prrevious_pregnancies', 'B_Previous_miscarriages', 'B_Previous_induced_abortions','B_Previous_ectopic_pregnancies',
          'B_Previous_births', 'B_stilborn','B_check_ups', 'B_check_ups_outpat','B_Number_of_fetuses','B_apgar_1m', 'B_apgar_5m']])


# In[ ]:


features.to_csv('/data/projects/project_avabalas/RNN/preprocessing_new/demographic_features.csv', index=False)


# In[ ]:


##################################################################


# In[ ]:


import pandas as pd
features = pd.read_csv('/data/projects/project_avabalas/RNN/preprocessing_new/demographic_features.csv')


# In[ ]:



from sklearn.preprocessing import MinMaxScaler
features = pd.read_csv('/data/projects/project_avabalas/RNN/preprocessing/demographic_features.csv')
scaler = MinMaxScaler()
features[['AGE', 'longitude_last', 'latitude_last','latitude_first', 'longitude_first', 'residence_duration_first','B_Mothers_age', 'B_Pregnancy_duration', 'B_Birth_weight','B_Birth length', 'Total_assistance',
          'Assistance_months','Assistance_years', 'SocHilm_Duration_days']] = scaler.fit_transform(features[['AGE', 'longitude_last', 'latitude_last','latitude_first', 'longitude_first', 'residence_duration_first',
          'B_Mothers_age', 'B_Pregnancy_duration', 'B_Birth_weight','B_Birth length', 'Total_assistance', 'Assistance_months','Assistance_years', 'SocHilm_Duration_days']])


# In[ ]:


features[['number_of_children', 'drug_purchases', 'kanta_prescriptions','B_prrevious_pregnancies', 'B_Previous_miscarriages', 'B_Previous_induced_abortions','B_Previous_ectopic_pregnancies', 'B_Previous_births', 'B_stilborn','B_check_ups', 'B_check_ups_outpat',
          'B_Number_of_fetuses','B_apgar_1m', 'B_apgar_5m']] = scaler.fit_transform(features[['number_of_children', 'drug_purchases', 'kanta_prescriptions','B_prrevious_pregnancies', 'B_Previous_miscarriages', 'B_Previous_induced_abortions','B_Previous_ectopic_pregnancies',
          'B_Previous_births', 'B_stilborn','B_check_ups', 'B_check_ups_outpat','B_Number_of_fetuses','B_apgar_1m', 'B_apgar_5m']])

features.to_csv('/data/projects/project_avabalas/RNN/preprocessing_new/demographic_features.csv', index=False)
# In[ ]:


features.max().values


# In[ ]:


features.iloc[:,24:35]


# In[ ]:





# In[ ]:


features = pd.read_csv('/data/projects/project_avabalas/RNN/preprocessing/demographic_features.csv')


# In[ ]:





# In[ ]:





# In[ ]:


dem = f.iloc[0,6:-2].astype(str).values.tolist()


# In[ ]:


' '.join(dem)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


continous.columns


# In[ ]:


continous.sha


# In[ ]:


ordinal.shape


# In[ ]:


binary.shape


# In[ ]:


one_h.shape


# In[ ]:


df


# In[ ]:


one_h


# In[ ]:


##########################################################################


# In[ ]:


import pandas as pd
import gc
import time
import datetime as dt
import numpy as np

path = '/data/processed_data/minimal_phenotype/minimal_phenotype_file.csv'
start_time = time.time()
df = pd.read_csv(path)
run_time = time.time()-start_time;print(run_time)


# In[ ]:


df['FINREGISTRYID'].nunique() # 96028 IDs are missing because of no sex information


# In[ ]:


duplicates = df.duplicated(subset=['FINREGISTRYID'])
df = df2[duplicates == False].copy() # 133 duplicates removed


# In[ ]:


df


# In[ ]:


df['date_of_birth'].isna().sum()


# In[ ]:


df['ever_divorced'].value_counts(dropna=False)


# In[ ]:


df['longitude_last'] = df['longitude_last'].fillna(df['longitude_last'].mean(skipna=True))
df['latitude_last'] = df['latitude_last'].fillna(df['latitude_last'].mean(skipna=True))
df['latitude_first'] = df['latitude_first'].fillna(df['latitude_first'].mean(skipna=True))
df['longitude_first'] = df['longitude_first'].fillna(df['longitude_first'].mean(skipna=True))

import datetime as dt
df['residence_start_date_first'] = pd.to_datetime(df['residence_start_date_first'] )
df['residence_end_date_first'] = pd.to_datetime(df['residence_end_date_first'] )
df['residence_duration_first'] = (df['residence_end_date_first'] - df['residence_start_date_first']).dt.days/365.24
df['residence_duration_first'] = df['residence_duration_first'].fillna(df['residence_duration_first'].mean(skipna=True))

df['Today'] = '2018-01-01'#date.today() 
df['Today'] = pd.to_datetime(df['Today'])
df['date_of_birth'] = pd.to_datetime(df['date_of_birth'])
df['AGE'] = (df['Today'] - df['date_of_birth']).dt.days/365.24


# In[ ]:


continous = df[['AGE','longitude_last','latitude_last','latitude_first','longitude_first',
                'residence_duration_first',]].copy()


# In[ ]:


df['received_social_assistance'] = df['received_social_assistance'].fillna(0)
df['assisted_living'] = df['assisted_living'].fillna(0)
df['probably_immigrated']= df['probably_immigrated'].fillna(0)
cat = df[['sex','received_social_assistance','assisted_living','probably_immigrated','index_person','possible_living_abroad']].copy()


# In[ ]:


df['residence_type_latest']=df['residence_type_latest'].fillna(4)
one_hot = pd.get_dummies(df['residence_type_latest'])
one_hot = one_hot.rename(columns={1.0: 'residence_with_DVV',2.0: 'residence_no_DVV',3.0: 'residence_foreign',4.0: 'residence_nan'})
cat2 = one_hot.copy()


# In[ ]:


df['residence_type_first']=df['residence_type_first'].fillna(4)
one_hot = pd.get_dummies(df['residence_type_first'])
one_hot = one_hot.rename(columns={1.0: 'residence_first_with_DVV',2.0: 'residence_first_no_DVV',3.0: 'residence_first_foreign',4.0: 'residence_first_nan'})
cat2 = cat2.join(one_hot)


# In[ ]:


df['mother_tongue']=df['mother_tongue'].fillna('unk')
one_hot = pd.get_dummies(df['mother_tongue'])
one_hot = one_hot.rename(columns={'fi': 'lang_fi','other': 'lang_other','ru': 'lang_ru','sv': 'lang_sv','unk': 'lang_unk'})
cat2 = cat2.join(one_hot)


# In[ ]:


df['ever_married']=df['ever_married'].fillna(2)
one_hot = pd.get_dummies(df['ever_married'])
one_hot = one_hot.rename(columns={0.0: 'ever_married_no',1.0: 'ever_married_yes',2.0: 'ever_married_nan'})
cat2 = cat2.join(one_hot)


# In[ ]:


df['ever_divorced']=df['ever_divorced'].fillna(2)
one_hot = pd.get_dummies(df['ever_divorced'])
one_hot = one_hot.rename(columns={0.0: 'ever_divorced_no',1.0: 'ever_divorced_yes',2.0: 'ever_divorced_nan'})
cat2 = cat2.join(one_hot)


# In[ ]:


features = df[['FINREGISTRYID']].copy()


# In[ ]:


features = features.join(continous)


# In[ ]:


features = features.join(cat)


# In[ ]:


features = features.join(cat2)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features[['AGE', 'longitude_last','latitude_last','latitude_first','longitude_first',
               'residence_duration_first']] = scaler.fit_transform(features[['AGE', 'longitude_last','latitude_last','latitude_first','longitude_first','residence_duration_first']])


# In[ ]:


features.to_csv('/data/project_avabalas/RNN/preprocessing/demographic_features.csv', index=False)


# In[ ]:


features = pd.read_csv('/data/project_avabalas/RNN/preprocessing/demographic_features.csv')


# In[ ]:


features['index_person'].value_counts()


# In[ ]:


features.shape


# In[ ]:





# In[ ]:


features['sex'].value_counts(dropna=False)


# In[ ]:


df['eduyears_ISCED97'].value_counts(dropna=False)


# In[ ]:


features.describe()


# In[ ]:


features.columns.values.tolist()


# In[ ]:


df['ever_divorced'].value_counts(dropna=False)


# In[ ]:


features.shape


# In[ ]:


features


# In[ ]:


df['AGE'].max()


# In[ ]:


cat2.iloc[:,10:]


# In[ ]:


continous.shape


# In[ ]:


cat2.shape


# In[ ]:





# In[ ]:


#sex,residence_type_last,residence_type_first,mother_tongue,ever_married,ever_divorced


# In[ ]:





# In[ ]:





# In[ ]:


df['sex'].value_counts(dropna=False)


# In[ ]:





# In[ ]:


# Get one hot encoding of columns B
one_hot = pd.get_dummies(df['B'])
# Drop column B as it is now encoded
df = df.drop('B',axis = 1)
# Join the encoded df
df = df.join(one_hot)
df  
Out[]:


# In[ ]:





# In[ ]:





# In[ ]:


one_hot = df[['received_social_assistance','assisted_living','probably_emigrated','probably_immigrated','index_person']].copy()


# In[ ]:


df['index_person'].value_counts(dropna=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


cat = df[['received_social_assistance','assisted_living','probably_emigrated','probably_immigrated','index_person']].copy()


# In[ ]:





# In[ ]:





# In[ ]:


continous


# In[ ]:


df['longitude_first'] = df['longitude_first'].fillna(df['longitude_first'].mean(skipna=True))


# In[ ]:


df['n_inpatient'].isna().value_counts()


# In[ ]:


df['Today'] = '2018-01-01'#date.today() 
df['Today'] = pd.to_datetime(df['Today'])
df['date_of_birth'] = pd.to_datetime(df['date_of_birth'])
df['AGE'] = (df['Today'] - df['date_of_birth']).dt.days/365.24


# In[ ]:


df['AGE'].isna().value_counts()


# In[ ]:


continous = df[['AGE','date_of_birth','longitude_last','latitude_last','latitude_first','longitude_first',
                'residence_duration_first',]].copy()


# In[ ]:


df['residence_end_date_first'].isna().value_counts()


# In[ ]:


#from datetime import date
import datetime as dt
df['residence_start_date_first'] = pd.to_datetime(df['residence_start_date_first'] )
df['residence_end_date_first'] = pd.to_datetime(df['residence_end_date_first'] )
df['residence_duration_first'] = (df['residence_end_date_first'] - df['residence_start_date_first']).dt.days/365.24
df['residence_duration_first'] = df['residence_duration_first'].fillna(df['residence_duration_first'].mean(skipna=True))


# In[ ]:


df['residence_duration_first'] = df['residence_duration_first'].fillna(df['residence_duration_first'].mean(skipna=True))


# In[ ]:


df['residence_duration_first'].mean()


# In[ ]:


dob = pd.read_csv('/data/processed_data/Finregistry_IDs_and_full_DOB.txt') #  'DOB(YYYY-MM-DD)'
cancer = cancer.merge(dob, on='FINREGISTRYID', how='left') # df_a.merge(df_b, on='mukey', how='left')

cancer['DOB(YYYY-MM-DD)'] = pd.to_datetime(cancer['DOB(YYYY-MM-DD)'])                             
cancer['EVENT_AGE'] = (df['dg_date'] - cancer['DOB(YYYY-MM-DD)']).dt.days/365.24
cancer.drop(columns='DOB(YYYY-MM-DD)', inplace=True)


# In[ ]:





# In[ ]:


df['longitude_last'] = df['longitude_last'].fillna(df['longitude_last'].mean(skipna=True))


# In[ ]:


df['longitude_last'].mean(skipna=True)


# In[ ]:


df.iloc[:,20:]


# In[ ]:


df.shape


# In[ ]:


df['latitude_last'].min()


# In[ ]:


df['FINREGISTRYID'].nunique()


# In[ ]:


df[df['FINREGISTRYID'].isna()]


# In[ ]:


df = df[df['FINREGISTRYID'].notna()]


# In[ ]:


df.shape


# In[ ]:


duplicates = df.duplicated(subset=['FINREGISTRYID'])
#nonduplicates = relatives[duplicates == False].copy()
#nonduplicates.rename(columns={"Relative_ID": "FINREGISTRYID"}, inplace=True)


# In[ ]:


duplicates[duplicates==True]


# In[ ]:


df.iloc[4976996,:]


# In[ ]:


df[df['FINREGISTRYID']=='FR0228634']


# In[ ]:


df = df.drop(4976996)


# In[ ]:


df.shape


# In[ ]:


features.to_csv('/data/project_avabalas/death/demographic_features_not_scaled.csv', index=False)


# In[ ]:




