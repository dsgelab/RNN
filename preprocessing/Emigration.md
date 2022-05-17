

```python
import pandas as pd
import gc
import time
import datetime as dt
import numpy as np

```


```python
lh = pd.read_csv('/data/processed_data/dvv/Tulokset_1900-2010_tutkhenk_asuinhist.txt.finreg_IDsp')
```


```python
rel = pd.read_csv('/data/processed_data/dvv/Tulokset_1900-2010_tutkhenk_ja_sukulaiset.txt.finreg_IDsp')
```


```python
lh = lh.sort_values(["FINREGISTRYID", "Start_of_residence"], ascending = (True, True))
```


```python
lh[(lh['Residence_type']==3)&(lh['End_of_residence'].isna())]['Municipality_name'].value_counts()
```


```python
lh_still_aborad = lh[(lh['Residence_type']==3)&(lh['End_of_residence'].isna())]
duplicates = lh_still_aborad.duplicated(subset=['FINREGISTRYID'])
lh_still_aborad= lh_still_aborad[duplicates == False].copy()
print(lh_still_aborad.shape[0])
```


```python
lh_still_aborad['live_abroad']=1
lh_still_aborad  = lh_still_aborad[['FINREGISTRYID','live_abroad']]
```


```python
lh_still_aborad
```


```python
emigrated = rel[rel['Emigration_date'].notna()]
```


```python
duplicates2 = emigrated.duplicated(subset=['Relative_ID'])
emigrated = emigrated[duplicates2 == False].copy()
```


```python
emigrated['live_abroad']=1
emigrated  = emigrated[['Relative_ID','live_abroad']]
```


```python
emigrated  = emigrated .rename(columns={'Relative_ID': 'FINREGISTRYID'})
```


```python
print("emigrated from DVV relatives",emigrated.shape[0], "lives aboread from DVV living history", lh_still_aborad.shape[0])
```


```python
emigrated = pd.concat([emigrated,lh_still_aborad])
```


```python
print(emigrated.shape[0])
duplicates3 = emigrated.duplicated(subset=['FINREGISTRYID'])
emigrated = emigrated[duplicates3 == False].copy()
print(emigrated.shape[0])
```


```python
emigrated.to_csv('/data/projects/project_avabalas/RNN/preprocessing_new/emigrated.csv',index=False)
```
