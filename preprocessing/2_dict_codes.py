#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import gc
import pandas as pd


# ## All codes with source, total code count, and count of IDs having a specific code

# In[ ]:


types={} 
for n in range(1,25):
    start_time = time.time()
    df = pd.read_csv('/data/projects/project_avabalas/RNN/preprocessing_new/combined_endp_atc.txt.'+str(n))
    df = df.groupby('FINREGISTRYID')
    
    for g, f in df:
    
        # count all codes per ID
        for code, source in zip(f['CODE'].values.tolist(),f['SOURCE'].values.tolist()):
            if code in types: types[code][2]=types[code][2]+1
            else: types[code] = [len(types)+1,source,1,0] 

        # Count unique codes per ID        
        f = f.drop_duplicates(subset = ["CODE"]) 
        for code in f['CODE'].values.tolist():
            types[code][3]=types[code][3]+1            
            
                
    run_time = time.time()-start_time;print(n,' ',run_time,' ', len(types))
    del df,f
    gc.collect() 


# In[ ]:


types_df = pd.DataFrame([(k, *v) for k, v in types.items()], columns=['Code','Token','Source','Count','Count_ID'])
print(types_df.shape[0])
types_df.to_csv('/data/projects/project_avabalas/RNN/preprocessing_new/code_dict.csv',index=False)