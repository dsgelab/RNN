

```python
import time
import pandas as pd
import numpy as np
```

## Delete omitted codes from the data


```python
endp = pd.read_excel('/home/avabalas/scripts/FINNGEN_ENDPOINTS_DF8_Final_2021-09-02_OMITS.xlsx')
```


```python
omits = endp[((endp['OMIT']==2) | (endp['OMIT']==1)) & (endp['Modification_reason']!='EXMORE/EXALLC priorities') & (endp['NAME']!='DEATH')]['NAME'].unique()
meds = endp[(endp['KELA_ATC'].notna()) & (endp['HD_ICD_10'].isna())]['NAME'].unique()
composite = endp[(endp['COD_ICD_10'].isna()) & (endp['HD_ICD_10'].isna()) & (endp['HD_ICD_10'].isna()) & (endp['CANC_TOPO'].isna())
                 & (endp['KELA_ATC'].isna()) & (endp['KELA_REIMB'].isna()) & (endp['OPER_NOM'].isna()) & ~(endp['NAME'].str.contains('#_This_follow'))]['NAME'].unique()
additional_omits = list(meds)+list(composite)
```


```python
new_omits = [item for item in additional_omits if item not in list(omits)] # will be faster than ussing additional_omits which contain already omited endpoints
```


```python
len(new_omits)
```


```python
dicty = pd.read_csv('/data/projects/project_avabalas/RNN/preprocessing_new/code_dict.csv')
```


```python
rare = dicty[dicty['Count_ID']<=70]['Code'].unique().tolist()
```


```python
len([item for item in rare if item not in new_omits] )
```


```python
new_omits = new_omits+rare
```


```python
for n in range(1,25):
    start_time = time.time()
    path_in = '/data/projects/project_avabalas/RNN/preprocessing_new/combined_endp_atc.txt.'+str(n)
    in_file = open(path_in)
    path_out = '/data/projects/project_avabalas/RNN/preprocessing_new/combined_endp_atc2.txt.'+str(n)
    out_file = open(path_out, "x")
    OUT_HEADER = in_file.readline()
    OUT_HEADER = ",".join(OUT_HEADER.rstrip("\n").split(','))
    print(OUT_HEADER, file=out_file)
    # write only not omited endpoints from a new list to an output file
    zzz = 0
    for row in in_file:
        records = row.rstrip("\n").split(',')
        if records[4] in new_omits:
            zzz = zzz+1
            continue            
        else:
            print(
                ",".join(records),
                file=out_file
            )
    in_file.close()
    out_file.close()
    run_time = time.time()-start_time;print(n,' ',run_time,'deleted rows:',zzz)    
```

## Delete omitted codes from the code dictionary 


```python
[item for item in new_omits if item not in dicty['Code'].values.tolist()]
```


```python
dicty.shape
```


```python
dicty = dicty[~dicty['Code'].isin(new_omits)]
```


```python
dicty.shape
```


```python
dicty = dicty.sort_values(["Source", "Count_ID"], ascending = (False, False))
```


```python
dicty['Token'] = np.arange(1,dicty.shape[0]+1)
```


```python
dicty.to_csv('/data/projects/project_avabalas/RNN/preprocessing_new/code_dict_no_rare_additional_omits.csv',index=False)
```
