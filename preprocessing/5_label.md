

```python
import pandas as pd
import numpy as np
import time
```


```python
data = pd.read_csv('/data/processed_data/dvv/Finregistry_IDs_and_full_DOB.txt') # read all IDs and DOB
```


```python
densified = pd.read_csv('/data/processed_data/endpointer/densified_first_events_DF8_subset_2021-09-04.txt') # for endpoint death
```


```python
# load a generated endpoint
densified = densified[densified['ENDPOINT']=='DEATH']
densified = densified[['FINNGENID','YEAR']]
densified = densified.rename(columns={'YEAR': 'ENDP'})
densified.rename(columns={"FINNGENID": 'FINREGISTRYID'}, inplace=True)
```


```python
# Loead death recorded in causes of death register
cod = pd.read_csv('/data/processed_data/sf_death/thl2019_1776_ksyy_tutkimus.csv.finreg_IDsp')
cod = cod.rename(columns={'TNRO': 'FINREGISTRYID'})
cod['COD'] = cod['KVUOSI']
cod = cod[['FINREGISTRYID','COD']]
```


```python
# Loead death recorded in DVV relatives register
relatives = pd.read_csv('/data/processed_data/dvv/Tulokset_1900-2010_tutkhenk_ja_sukulaiset.txt.finreg_IDsp', usecols = ['Relative_ID','Relative_death_date'])
relatives = relatives[relatives['Relative_ID'].notna()]
duplicates = relatives.duplicated(subset=['Relative_ID'])
relatives = relatives[duplicates == False].copy()
relatives.rename(columns={"Relative_ID": "FINREGISTRYID"}, inplace=True)
relatives = relatives[relatives['Relative_death_date'].notna()].copy()
relatives.rename(columns={"Relative_death_date": "DVV"}, inplace=True)
relatives['DVV']=relatives['DVV'].apply(lambda x : x[:4])
relatives['DVV'] = relatives['DVV'].astype(int)
relatives = relatives[relatives['DVV']!=2020] # exclude 20190 deaths in 2020
```


```python
# Merge death recorded from differenc sources
print(data.shape[0])
data = data.merge(densified, on='FINREGISTRYID', how='left') 
data = data.merge(cod, on='FINREGISTRYID', how='left') 
data = data.merge(relatives, on='FINREGISTRYID', how='left') 
print(data[data['ENDP'].notna()].shape[0]==densified.shape[0],data[data['COD'].notna()].shape[0]==cod.shape[0],data[data['DVV'].notna()].shape[0]==relatives.shape[0],data.shape[0])
```


```python
data['DEATH_YEAR'] = 0
```


```python
#data[(data['ENDP'].notna())|(data['COD'].notna())|(data['DVV'].notna())]
print('The number of COD recorded but DVV not recorded deaths', data.loc[(data['COD'].notna())&(data['DVV'].isna()),'DEATH_YEAR'].shape[0])
print('The number of DVV recorded but COD not recorded deaths', data.loc[(data['COD'].isna())&(data['DVV'].notna()),'DEATH_YEAR'].shape[0])
print('The number of deaths recorded in both registers', data.loc[(data['COD'].notna())&(data['DVV'].notna()),'DEATH_YEAR'].shape[0])
#The number of COD recorded but DVV not recorded deaths 46283
#The number of DVV recorded but COD not recorded deaths 34109
#The number of deaths recorded in both registers 1512025
```


```python
data.loc[(data['COD'].notna())&(data['DVV'].isna()),'DEATH_YEAR'] = data.loc[(data['COD'].notna())&(data['DVV'].isna()),'COD']
data.loc[(data['COD'].isna())&(data['DVV'].notna()),'DEATH_YEAR'] = data.loc[(data['COD'].isna())&(data['DVV'].notna()),'DVV']
data.loc[(data['COD'].notna())&(data['DVV'].notna()),'DEATH_YEAR'] = data.loc[(data['COD'].notna())&(data['DVV'].notna()),'DVV']
print('The total number of deaths', data[data['DEATH_YEAR']!=0].shape[0]) # The total number of deaths 1592417
```


```python
data['DEATH']=0
data['DEATH_YEAR'] = data['DEATH_YEAR'].astype(int)
```


```python
data.loc[(data['DEATH_YEAR']>2017),'DEATH'] = 1 # cases
```


```python
data.loc[(data['DEATH_YEAR']<=2017)&(data['DEATH_YEAR']!=0),'DEATH'] = 2 # neither cases nor controls controls - individuals who dies befre 2019
```


```python
print(data['DEATH'].value_counts())
print('case control ratio:',data[data['DEATH']==1].shape[0]/data[data['DEATH']==0].shape[0]*100 )

# 0    5573999
# 2    1483115
# 1     109302
```


```python
data = data[['FINREGISTRYID','DEATH']]
data = data.rename(columns={'DEATH': 'LABEL'})
```


```python
data.to_csv('/data/projects/project_avabalas/RNN/preprocessing_new/label.csv', index = False)
```