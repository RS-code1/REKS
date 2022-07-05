import pandas as pd
import numpy as np

path=''

# ##user->purchase->product
n=0
cols=['uid','pid','bid','cid','ab','av','bt','oc']
user_product=pd.DataFrame([],columns=cols)
file = open(path+'./data/train.txt')
for line in file:
    if n>=1:
        user_id=line.replace('\n','').strip().split('\t')[0]
        item_id = line.replace('\n', '').strip().split('\t')[1]
        idxs = [-1] * 8
        idxs[0]=int(user_id)
        idxs[1]=int(item_id)
        user_product=user_product.append(dict(zip(cols,idxs)),ignore_index=True)
    n+=1

print('finished1')

# product + produced_by -> brand
n=0
p_p_b=pd.DataFrame([],columns=cols)
file = open(path+'./data/brand_p_b.txt')
for line in file:
    idxs=[-1]*8
    if n>=1:
        brand=line.replace('\n', '').strip()
        if len(brand) > 0 and brand!= '""':
            idxs = [-1] * 8
            idxs[1] = n - 1
            idxs[2] = int(brand)
            p_p_b = p_p_b.append(dict(zip(cols, idxs)), ignore_index=True)
    n+=1
print('finished2')

# product + belongs_to -> category
n=0
p_b_c=pd.DataFrame([],columns=cols)
file = open(path+'./data/category_p_c.txt')
for line in file:
    if n>=1:
        categories=line.replace('\n', '').strip().split(' ')
        for x in categories:
            if len(x) > 0 and x != '""':
                idxs=[-1]*8
                idxs[1]=n-1
                idxs[3]=int(x)
                p_b_c=p_b_c.append(dict(zip(cols,idxs)),ignore_index=True)
    n+=1
print('finished3')

 # product + also_bought -> related_product
n=0
p_a_b=pd.DataFrame([],columns=cols)
file = open(path+'./data/also_bought_p_p.txt')
for line in file:
    if n>=1:
        r_products=line.replace('\n', '').strip().split(' ')
        for x in r_products:
            if len(x) > 0 and x != '""':
                idxs=[-1]*8
                idxs[1]=n-1
                idxs[4]=int(x)
                p_a_b=p_a_b.append(dict(zip(cols,idxs)),ignore_index=True)
    n+=1
print('finished4')

# product + also_viewed -> related_product
n=0
p_a_v=pd.DataFrame([],columns=cols)
file = open(path+'./data/also_viewed_p_p.txt')
for line in file:
    if n>=1:
        r_products=line.replace('\n', '').strip().split(' ')
        for x in r_products:
            if len(x) > 0 and x != '""':
                idxs=[-1]*8
                idxs[1]=n-1
                idxs[5]=int(x)
                p_a_v=p_a_v.append(dict(zip(cols,idxs)),ignore_index=True)
    n+=1
print('finished5')

# product + also_bought -> related_product
n=0
p_b_t=pd.DataFrame([],columns=cols)
file = open(path+'./data/bought_together_p_p.txt')
for line in file:
    if n>=1:
        r_products=line.replace('\n', '').strip().split(' ')
        for x in r_products:
            if len(x) > 0 and x != '""':
                idxs=[-1]*8
                idxs[1]=n-1
                idxs[6]=int(x)
                p_b_t=p_b_t.append(dict(zip(cols,idxs)),ignore_index=True)
    n+=1
print('finished5')

# product + co_occr -> product
n=0
p_o_c=pd.DataFrame([],columns=cols)
file = open(path+'./data/co_occr_p_p.txt')
for line in file:
    if n>=1:
        r_products=line.replace('\n', '').strip().split(' ')
        for x in r_products:
            if len(x) > 0 and x != '""':
                idxs=[-1]*8
                idxs[1]=n-1
                idxs[7]=int(x)
                p_o_c=p_o_c.append(dict(zip(cols,idxs)),ignore_index=True)
    n+=1
print('finished6')


transe_train_data=pd.concat([user_product,p_p_b,p_b_c,p_a_b,p_a_v,p_b_t,p_o_c],axis=0)
transe_train_data.to_csv(path+'./data/transe_train_data_co.csv',index=None)
#df = pd.read_csv('transe_train_data.csv')
#dataset_trans=df.drop(df[df.oc != -1].index)

#train_data=pd.concat([dataset_trans,p_o_c],axis=0)
#train_data.to_csv(path+'transe_train_data_co.csv',index=None)






