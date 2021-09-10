# pip install xlsxwriter
# pip install streamlit
# pip install pandas
# pip install requests
# pip install google-search-results
# pip install py_stringmatching

# run the file :
# streamlit run st.py 

from typing import Text
import pandas as pd
import requests
import streamlit as st
import time 
from serpapi import GoogleSearch
import numpy as np
import py_stringmatching as sm
import base64
from io import BytesIO

api_key = 'e478f284e8aa736bc21fd8691ae7d08f14680d2e6a1fac7a8d6ad1f51e1b358f'


data = st.text_area('Add the Keywords and press the button')

result = []






def serch_keyword(key) :
    params = {
    "engine": "google",
    "q": key,
    "num": "30",
    "api_key": api_key
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results['organic_results']
    try :
        total_results = results['search_information']['total_results']
    except:
        total_results = 0
    return total_results,organic_results
records = []
def dati_clc (result) :
    for key in result :
        total_results,organic_results = serch_keyword(key)
        for r in organic_results :
            rank = r['position']
            title = r['title']
            url = r['link']

            records.append({
                'keyword' : key,
                'rank' : rank,
                'url' : url,
                'search_volume' :total_results,
            })
if st.button('Start Process The Keyword'):
    tt =data.splitlines()
    dati_clc(tt)

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df,name):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{name}.xlsx">Download Excel file</a>' # decode b'abc' => abc



# Split 
serps = pd.DataFrame(records)

st.dataframe(serps)
st.markdown(get_table_download_link(serps,'serp'), unsafe_allow_html=True)

try :

    serps_grpby_keyword = serps.groupby("keyword")
    k_urls = 15
except :
    pass
# Apply Combine
def filter_k_urls(group_df):
    filtered_df = group_df.loc[group_df['url'].notnull()]
    filtered_df = filtered_df.loc[filtered_df['rank'] <= k_urls]
    return filtered_df
try :
    filtered_serps = serps_grpby_keyword.apply(filter_k_urls)
except Exception as e :
    print(e)
    pass
# Combine
## Add prefix to column names
#normed = normed.add_prefix('normed_')
try :
    # Concatenate with initial data frame
    filtered_serps_df = pd.concat([filtered_serps],axis=0)
    del filtered_serps_df['keyword']
    filtered_serps_df = filtered_serps_df.reset_index()
    del filtered_serps_df['level_1']


    # convert results to strings using Split Apply Combine
    filtserps_grpby_keyword = filtered_serps_df.groupby("keyword")
except Exception as e :
    print(e)
    pass
def string_serps(df):
    df['serp_string'] = ''.join(df['url'])
    return df    
try:
    # Combine
    strung_serps = filtserps_grpby_keyword.apply(string_serps)

    # Concatenate with initial data frame and clean
    strung_serps = pd.concat([strung_serps],axis=0)
    strung_serps = strung_serps[['keyword', 'serp_string']]#.head(30)
    strung_serps = strung_serps.drop_duplicates()
except Exception as e :
    print(e)
    pass
# align serps
def serps_align(k, df):
    prime_df = df.loc[df.keyword == k]
    prime_df = prime_df.rename(columns = {"serp_string" : "serp_string_a", 'keyword': 'keyword_a'})
    comp_df = df.loc[df.keyword != k].reset_index(drop=True)
    prime_df = prime_df.loc[prime_df.index.repeat(len(comp_df.index))].reset_index(drop=True)
    prime_df = pd.concat([prime_df, comp_df], axis=1)
    prime_df = prime_df.rename(columns = {"serp_string" : "serp_string_b", 'keyword': 'keyword_b', "serp_string_a" : "serp_string", 'keyword_a': 'keyword'})
    return prime_df
try:
    columns = ['keyword', 'serp_string', 'keyword_b', 'serp_string_b']
    matched_serps = pd.DataFrame(columns=columns)
    matched_serps = matched_serps.fillna(0)
    queries = strung_serps.keyword.to_list()

    for q in queries:
        temp_df = serps_align(q, strung_serps)
        matched_serps = matched_serps.append(temp_df)
    ws_tok = sm.WhitespaceTokenizer()
except Exception as e :
    print(e)
    pass


# Only compare the top k_urls results 
def serps_similarity(serps_str1, serps_str2, k=15):
    denom = k+1
    norm = sum([2*(1/i - 1.0/(denom)) for i in range(1, denom)])

    ws_tok = sm.WhitespaceTokenizer()

    serps_1 = ws_tok.tokenize(serps_str1)[:k]
    serps_2 = ws_tok.tokenize(serps_str2)[:k]

    match = lambda a, b: [b.index(x)+1 if x in b else None for x in a]

    pos_intersections = [(i+1,j) for i,j in enumerate(match(serps_1, serps_2)) if j is not None] 
    pos_in1_not_in2 = [i+1 for i,j in enumerate(match(serps_1, serps_2)) if j is None]
    pos_in2_not_in1 = [i+1 for i,j in enumerate(match(serps_2, serps_1)) if j is None]
    a_sum = sum([abs(1/i -1/j) for i,j in pos_intersections])
    b_sum = sum([abs(1/i -1/denom) for i in pos_in1_not_in2])
    c_sum = sum([abs(1/i -1/denom) for i in pos_in2_not_in1])

    intent_prime = a_sum + b_sum + c_sum
    intent_dist = 1 - (intent_prime/norm)
    return intent_dist
# Apply the function
try:
    matched_serps['si_simi'] = matched_serps.apply(lambda x: serps_similarity(x.serp_string, x.serp_string_b), axis=1)
    serps_compared = matched_serps[['keyword', 'keyword_b', 'si_simi']]

    # group keywords by search intent
    simi_lim = 0.4

    # join search volume
    keysv_df = serps[['keyword', 'search_volume']].drop_duplicates()
    keysv_df.head()

    # append topic vols
    keywords_crossed_vols = serps_compared.merge(keysv_df, on = 'keyword', how = 'left')
    keywords_crossed_vols = keywords_crossed_vols.rename(columns = {'keyword': 'topic', 'keyword_b': 'keyword',
                                                                    'search_volume': 'topic_volume'})

    # sim si_simi
    keywords_crossed_vols.sort_values('topic_volume', ascending = False)


    # strip NANs
    keywords_filtered_nonnan = keywords_crossed_vols.dropna()

    st.dataframe(keywords_filtered_nonnan)
    st.markdown(get_table_download_link(keywords_filtered_nonnan,'keywords_filtered_nonnan'), unsafe_allow_html=True)
    queries_in_df = list(set(keywords_filtered_nonnan.topic.to_list()))
except Exception as e :
    print(e)
    pass
topic_groups_numbered = {}
topics_added = []

def find_topics(si, keyw, topc):
    i = 0
    if (si >= simi_lim) and (not keyw in topics_added) and (not topc in topics_added): 
        i += 1     
        topics_added.append(keyw)
        topics_added.append(topc)
        topic_groups_numbered[i] = [keyw, topc]          
    elif si >= simi_lim and (keyw in topics_added) and (not topc in topics_added):  
        j = [key for key, value in topic_groups_numbered.items() if keyw in value]
        topics_added.append(topc)
        topic_groups_numbered[j[0]].append(topc)

    elif si >= simi_lim and (not keyw in topics_added) and (topc in topics_added):
        j = [key for key, value in topic_groups_numbered.items() if topc in value]        
        topics_added.append(keyw)
        topic_groups_numbered[j[0]].append(keyw) 

def apply_impl_ft(df):
    return df.apply(
        lambda row:
            find_topics(row.si_simi, row.keyword, row.topic), axis=1)
try:
    apply_impl_ft(keywords_filtered_nonnan)

    topic_groups_numbered = {k:list(set(v)) for k, v in topic_groups_numbered.items()}

    topic_groups_lst = []

    for k, l in topic_groups_numbered.items():
        for v in l:
            topic_groups_lst.append([k, v])

    topic_groups_dictdf = pd.DataFrame(topic_groups_lst, columns=['topic_group_no', 'keyword'])
    st.dataframe(topic_groups_dictdf)
    st.markdown(get_table_download_link(topic_groups_dictdf,'topic_groups_dictdf'), unsafe_allow_html=True)

except Exception as e :
    print(e)
    pass
