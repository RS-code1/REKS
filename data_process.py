import pandas as pd
import numpy as np
import gzip
import string
import re

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF('./data/reviews_Beauty_5.json.gz')

def make_sessions(data, session_th=24 * 60 * 60, is_ordered=False, user_key='reviewerID', item_key='asin', time_key='unixReviewTime'):
    """Assigns session ids to the events in data without grouping keys"""
    if not is_ordered:
        # sort data by user and time
        data.sort_values(by=[user_key, time_key], ascending=True, inplace=True)
    # compute the time difference between queries
    tdiff = np.diff(data[time_key].values)
    # check which of them are bigger then session_th
    split_session = tdiff > session_th
    split_session = np.r_[True, split_session]
    # check when the user chenges is data
    new_user = data[user_key].values[1:] != data[user_key].values[:-1]
    new_user = np.r_[True, new_user]
    # a new sessions stars when at least one of the two conditions is verified
    new_session = np.logical_or(new_user, split_session)
    # compute the session ids
    session_ids = np.cumsum(new_session)
    data['session_id'] = session_ids
    return data

def last_session_out_split(data,
                            user_key='reviewerID',
                            item_key='asin',
                            session_key='session_id',
                            time_key='unixReviewTime',
                            clean_test=True,
                            min_session_length=2):
    """
    last-session-out split
    assign the last session of every user to the test set and the remaining ones to the training set
    """
    sessions = data.sort_values(by=[user_key, time_key]).groupby(user_key)[session_key]
    last_session = sessions.last()
    train = data[~data.session_id.isin(last_session.values)].copy()
    slen = train[session_key].value_counts()
    good_sessions = slen[slen >= min_session_length].index
    train = train[train[session_key].isin(good_sessions)].copy()
    test = data[data.session_id.isin(last_session.values)].copy()
    if clean_test:
        train_users = train[user_key].unique()
        test = test[test[user_key].isin(train_users)]
        train_items = train[item_key].unique()
        test = test[test[item_key].isin(train_items)]
        #  remove sessions in test shorter than min_session_length
        #slen = test[session_key].value_counts()
        #good_sessions = slen[slen >= min_session_length].index
        #test = test[test[session_key].isin(good_sessions)].copy()
    return train, test

data=make_sessions(df)

#print(data)
train,test=last_session_out_split(data)


users=train['reviewerID'].unique()
user_dic=dict(zip(users,np.arange(len(users))))
np.save('./data/user_dict.npy',user_dic)

items=train['asin'].unique()

item_dic=dict(zip(items,np.arange(len(items))))

np.save('./data/item_dict.npy',item_dic)

train['review']=train['reviewText'].apply(lambda x: re.sub(r'[{}]+'.format(string.punctuation),'',x).lower())
train['review_word']=train['review'].apply(lambda x: list(x.split(' ')))
all_review=train['review_word'].tolist()
all_word=[]
for x in all_review:
  all_word+=x

word_list=list(set(all_word))
word_list=[x for x in word_list if x not in ['']]
word_dic=dict(zip(word_list,np.arange(len(word_list))))
np.save('./data/word_dict.npy',word_dic)

df_word_dic=pd.DataFrame({'word':list(word_dic.keys()),'word_id':list(word_dic.values())})
df_word_dic.sort_values(by=['word_id'],ascending=True,inplace=True)
df_word_dic[['word']].to_csv('./data/vocab.txt',index=False)






train['user_id']=train['reviewerID'].apply(lambda x: user_dic[x])
test['user_id']=test['reviewerID'].apply(lambda x: user_dic[x])
train['item_id']=train['asin'].apply(lambda x: item_dic[x])
test['item_id']=test['asin'].apply(lambda x: item_dic[x])

def trans(review,dic):
  return [dic[x] for x in review if x not in ['']]

train['review_word']=train['review_word'].apply(lambda x : trans(x,word_dic))
#test['review_word']=test['review_word'].apply(lambda x : trans(x,word_dic))
print("aaaaaaaaaaaaaa",len(train['asin'].unique()))
train.to_csv('./data/train.csv')
test.to_csv('./data/test.csv')


meta = getDF('./data/meta_Beauty.json.gz')
##product.txt
items_dic=np.load('./data/item_dict.npy',allow_pickle=True).item()
items=items_dic.keys()

df_item_dic=pd.DataFrame({'asin':list(items),'item_id':list(items_dic.values())})
df_item_dic.sort_values(by=['item_id'],ascending=True,inplace=True)
df_item_dic[['asin']].to_csv('./data/product.txt',index=None)

##users.txt
user_dic=np.load('./data/user_dict.npy',allow_pickle=True).item()
users=user_dic.keys()
df_user_dic=pd.DataFrame({'user':list(users),'user_id':list(user_dic.values())})
df_user_dic.sort_values(by=['user_id'],ascending=True,inplace=True)
df_user_dic[['user']].to_csv('./data/users.txt',index=None)

##brands.txt
meta=meta[meta.asin.isin(items)]
brands=list(meta['brand'].unique())
brands.remove(np.nan)
brand_dic=dict(zip(brands,np.arange(len(brands))))
np.save('./data/brand_dict.npy',brand_dic)
df_brand_dic=pd.DataFrame({'brand':list(brand_dic.keys()),'brand_id':list(brand_dic.values())})
df_brand_dic.sort_values(by=['brand_id'],ascending=True,inplace=True)
df_brand_dic[['brand']].to_csv('./data/brands.txt',index=None)

## brand_b_p.txt
brand_pb=dict(zip(items,['' for _  in range(len(items))]))
meta_brand=meta['brand'].tolist()
asin=meta['asin'].tolist()
for i in range(len(asin)):
    if meta_brand[i] not in brands:
        continue
    brand_pb[asin[i]]=meta_brand[i]

brand_dic['']=''
df_brand=pd.DataFrame({'asin':list(brand_pb.keys()),'brand':list(brand_pb.values())})
df_brand['item_id']=df_brand['asin'].apply(lambda x :items_dic[x])
df_brand['brand_id']=df_brand['brand'].apply(lambda x :brand_dic[x])
df_brand.sort_values(by=['item_id'],ascending=True,inplace=True)
df_brand[['brand_id']].to_csv('./data/brand_p_b.txt',index=None)


related=meta['related'].tolist()
asin=meta['asin'].tolist()

also_bought=dict(zip(items,[[] for _  in range(len(items))]))
also_viewed=dict(zip(items,[[] for _  in range(len(items))]))
bought_together=dict(zip(items,[[] for _  in range(len(items))]))
related_product=[]

for i in range(len(asin)):
    try:
        if 'also_bought' in related[i]:
            also_bought[asin[i]]+=related[i]['also_bought']
            related_product+=related[i]['also_bought']
        if 'also_viewed' in related[i]:
            also_viewed[asin[i]] += related[i]['also_viewed']
            related_product += related[i]['also_viewed']
        if 'bought_together' in related[i]:
            bought_together[asin[i]]+=related[i]['bought_together']
            related_product += related[i]['bought_together']
    except:
        continue

### categories.txt
meta_cate=meta['categories'].tolist()
categories=[]
cate_pc=dict(zip(items,[[] for _  in range(len(items))]))
for i in range(len(asin)):
    try:
        ca=list(np.array(meta_cate[i]).flat)
        ca=list(set(ca))
        cate_pc[asin[i]]=ca
        categories+=ca
    except:
        continue


categories=list(set(categories))
cate_dic=dict(zip(categories,np.arange(len(categories))))
df_cate_dic=pd.DataFrame({'category':list(cate_dic.keys()),'cate_id':list(cate_dic.values())})
df_cate_dic.sort_values(by=['cate_id'],ascending=True,inplace=True)
df_cate_dic[['category']].to_csv('./data/categories.txt',index=None)

###category_b_p.txt
df_cate_pc=pd.DataFrame({'item_id':list(cate_pc.keys()),'cate':list(cate_pc.values())})
def trans_cate(cate,dic):
    if len(cate)>=1:
        return [str(dic[x]) for x in cate ]
    return []


df_cate_pc['cate_lis']=df_cate_pc['cate'].apply(lambda x: ' '.join(trans_cate(x,cate_dic)))
df_cate_pc[['cate_lis']].to_csv('./data/category_p_c.txt',index=None)

###train.txt
train=pd.read_csv('./data/train.csv')
review=train[['user_id','item_id','review_word']]
review['review_word_str']=review['review_word'].apply(lambda x : (' ').join('%s'%id for id in eval(x)))
del review['review_word']
review.to_csv('./data/train.txt',sep='\t',index=None)

###test.txt
test=pd.read_csv('./data/test.csv')
review=test[['user_id','item_id']]
review.to_csv('./data/test.txt',sep='\t',index=None)


###NARM sample
train = pd.read_csv('./data/train.csv')
test=pd.read_csv('./data/test.csv')
maxlen=5
sample_train = []
label_train = []
user_train=[]
sess_train = train['session_id'].unique()
for se in sess_train:
    item_list = list(train[train['session_id'] == se]['item_id'])
    uid=train[train['session_id'] == se]['user_id'].unique()[0]
    for i in range(1, len(item_list)):
        if i <= maxlen:
            sample_item = item_list[:i]
            sample_train.append(sample_item)
            label_train.append(item_list[i])
            user_train.append(uid)
        else:
            break

sample_test = []
label_test = []
user_test=[]
sess_test = test['session_id'].unique()
for se in sess_test:
    item_list = list(test[test['session_id'] == se]['item_id'])
    uid = test[test['session_id'] == se]['user_id'].unique()[0]
    for i in range(1, len(item_list)):
        if i <= maxlen:
            sample_item = item_list[:i]
            sample_test.append(sample_item)
            label_test.append(item_list[i])
            user_test.append(uid)
        else:
            break

train_session=pd.DataFrame({'session':sample_train,'label':label_train,'user':user_train})
test_session=pd.DataFrame({'session':sample_test,'label':label_test,'user':user_test})
train_session.to_csv('./data/train_session.csv',index=None)
test_session.to_csv('./data/test_session.csv',index=None)



n_items=len(item_dic)


sessions=train_session['session'].tolist()

co_occr=dict(zip(np.arange(n_items),[[] for _ in range(n_items)]))
item_dict=dict(zip(list(item_dic.values()),list(item_dic.keys())))

for se in sessions:
    for i in range(len(se)-1):
        co_occr[se[i]].extend([se[i+1]])



def trans_related(lis,dic):
    if len(lis)>=1:
        return [str(dic[x]) for x in lis]
    return []


##related_product.txt
#related_product+=co_product
related_product=list(set(related_product))
related_product_dic=dict(zip(related_product,np.arange(len(related_product))))
df_rp_dic=pd.DataFrame({'related_product':list(related_product_dic.keys()),'related_product_id':list(related_product_dic.values())})
df_rp_dic.sort_values(by=['related_product_id'],ascending=True,inplace=True)
df_rp_dic[['related_product']].to_csv('./data/related_product.txt',index=None)

##co_occr_p_p.txt
coo=list(co_occr.values())
for i in range(len(coo)):
    coo[i]=list(set(coo[i]))
    #coo[i]=" ".join(str(i) for i in coo
    
co_occr_pd=pd.DataFrame({'item_id':list(co_occr.keys()),'related_product':coo})
for i in range(co_occr_pd.shape[0]):
    co_occr_pd.iloc[i,1]=" ".join(str(j) for j in coo[i])
co_occr_pd.sort_values(by=['item_id'],ascending=True,inplace=True)
co_occr_pd[['related_product']].to_csv('./data/co_occr_p_p.txt',index=None)


##also_bought_p_p.txt
df_also_bought=pd.DataFrame({'item':list(also_bought.keys()),'related_product':list(also_bought.values())})
df_also_bought['item_id']=df_also_bought['item'].apply(lambda x: items_dic[x])
df_also_bought['related_product_lis']=df_also_bought['related_product'].apply(lambda x :' '.join(trans_related(x,related_product_dic)))
df_also_bought.sort_values(by=['item_id'],ascending=True,inplace=True)
df_also_bought[['related_product_lis']].to_csv('./data/also_bought_p_p.txt',index=None)

##also_viewed_p_p.txt
df_also_viewed=pd.DataFrame({'item':list(also_viewed.keys()),'related_product':list(also_viewed.values())})
df_also_viewed['item_id']=df_also_viewed['item'].apply(lambda x: items_dic[x])
df_also_viewed['related_product_lis']=df_also_viewed['related_product'].apply(lambda x :' '.join(trans_related(x,related_product_dic)))
df_also_viewed.sort_values(by=['item_id'],ascending=True,inplace=True)
df_also_viewed[['related_product_lis']].to_csv('./data/also_viewed_p_p.txt',index=None)

##bought_together_p_p.txt
df_bought_together=pd.DataFrame({'item':list(bought_together.keys()),'related_product':list(bought_together.values())})
df_bought_together['item_id']=df_bought_together['item'].apply(lambda x: items_dic[x])
df_bought_together['related_product_lis']=df_bought_together['related_product'].apply(lambda x :' '.join(trans_related(x,related_product_dic)))
df_bought_together.sort_values(by=['item_id'],ascending=True,inplace=True)
df_bought_together[['related_product_lis']].to_csv('./data/bought_together_p_p.txt',index=None)

