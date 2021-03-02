import pandas as pd
import numpy as np
import gc
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from tqdm.notebook import tqdm
import lightgbm as lgb
from statistics import harmonic_mean 
from scipy.stats.mstats import hmean
train_pickle ='R:/Yann/riiid/cv1_train.pickle'
valid_pickle ='R:/Yann/riiid/cv1_valid.pickle'
question_file = 'R:/Yann/riiid/questions.csv'
debug = False
validaten_flg = False

# funcs for user stats with loop
def add_user_feats(df, answered_correctly_sum_u_dict, count_u_dict):
    acsu = np.zeros(len(df), dtype=np.int32)
    cu = np.zeros(len(df), dtype=np.int32)
    for cnt,row in enumerate(tqdm(df[['user_id','answered_correctly']].values)):
        acsu[cnt] = answered_correctly_sum_u_dict[row[0]]
        cu[cnt] = count_u_dict[row[0]]
        answered_correctly_sum_u_dict[row[0]] += row[1]
        count_u_dict[row[0]] += 1
    user_feats_df = pd.DataFrame({'answered_correctly_sum_u':acsu, 'count_u':cu})
    user_feats_df['answered_correctly_avg_u'] = user_feats_df['answered_correctly_sum_u'] / user_feats_df['count_u']
    df = pd.concat([df, user_feats_df], axis=1)
    return df

def add_user_feats_without_update(df, answered_correctly_sum_u_dict, count_u_dict):
    acsu = np.zeros(len(df), dtype=np.int32)
    cu = np.zeros(len(df), dtype=np.int32)
    for cnt,row in enumerate(df[['user_id']].values):
        acsu[cnt] = answered_correctly_sum_u_dict[row[0]]
        cu[cnt] = count_u_dict[row[0]]
    user_feats_df = pd.DataFrame({'answered_correctly_sum_u':acsu, 'count_u':cu})
    user_feats_df['answered_correctly_avg_u'] = user_feats_df['answered_correctly_sum_u'] / user_feats_df['count_u']
    df = pd.concat([df, user_feats_df], axis=1)
    return df

def update_user_feats(df, answered_correctly_sum_u_dict, count_u_dict):
    for row in df[['user_id','answered_correctly','content_type_id']].values:
        if row[2] == 0:
            answered_correctly_sum_u_dict[row[0]] += row[1]
            count_u_dict[row[0]] += 1

    state = defaultdict(dict)
    for user_id in np.sort(train['user_id'].unique()):
        state[user_id] = {}
    total = len(state.keys())
    user_content = train.groupby('user_id')['content_id'].apply(np.array).apply(np.sort).apply(np.unique)
    user_attempts = train.groupby(['user_id', 'content_id'])['content_id'].count().astype(np.uint8).groupby('user_id').apply(np.array).values
    user_attempts -= 1
    for user_id, content, attempt in tqdm(zip(state.keys(), user_content, user_attempts),total=total):
        state[user_id] = dict(zip(content, attempt))
    del user_content, user_attempts
    gc.collect()
    
    return state

def update_state(state, test_df):
    attempt = []
    for idx, (a, b) in test_df[['user_id', 'content_id']].iterrows():
        # check if user exists
        if a in state:
            # check if user already answered the question, if so update it to a maximum of 10
            if b in state[a]:
                num = (str(state[a][b])).replace("[","")
                num = num.replace("]","")
                state[a][b] = min(4,int(num)+1)
            # if user did not answered the question already, set the number of attempts to 0
            else:
                state[a][b] = 0
        else:
            state[a] =  dict(zip([b],[0]))
        # add user data to lists
        attempt.append(state[a][b])
    
    return state, attempt

feld_needed = ['row_id', 'user_id', 'content_id', 'content_type_id', 'answered_correctly', 'prior_question_elapsed_time', 'prior_question_had_explanation']
train = pd.read_pickle(train_pickle)[feld_needed]
#train = train[:9600000]
valid = pd.read_pickle(valid_pickle)[feld_needed]
#valid = valid[:1200000]
if debug:
    train = train[:6000000]
    valid = valid[:60000]
train = train.loc[train.content_type_id == False].reset_index(drop=True)
valid = valid.loc[valid.content_type_id == False].reset_index(drop=True)

# Attempt No
train["attempt"] = 1
train["attempt"] = train[["user_id","content_id","attempt"]].groupby(["user_id","content_id"])["attempt"].cumsum()
train['attempt'].values[train['attempt'].values > 4] = 4


questions_df = pd.read_csv(question_file)
state = get_state()
state, attempt = update_state(state, valid)
valid["attempt"] = [i+1 for i in attempt]
del attempt
gc.collect()


# answered correctly average for each content
content_df = train[['content_id','answered_correctly']].groupby(['content_id']).agg(['mean']).reset_index()
content_df.columns = ['content_id', 'answered_correctly_avg_c']
train = pd.merge(train, content_df, on=['content_id'], how="left")


# user stats features with loops
answered_correctly_sum_u_dict = defaultdict(int)
count_u_dict = defaultdict(int)
train = add_user_feats(train, answered_correctly_sum_u_dict, count_u_dict)
valid = add_user_feats(valid, answered_correctly_sum_u_dict, count_u_dict)
valid = pd.merge(valid, content_df, on=['content_id'], how="left")


# fill with mean value for prior_question_elapsed_time
# note that `train.prior_question_elapsed_time.mean()` dose not work!
# please refer https://www.kaggle.com/its7171/can-we-trust-pandas-mean for detail.
prior_question_elapsed_time_mean = train.prior_question_elapsed_time.dropna().values.mean()
train['prior_question_elapsed_time'] = train.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean)
valid['prior_question_elapsed_time'] = valid.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean)

# Calculate Harmonic Mean between User and Content Avgs
train['hmean'] = 2 / ((1/train.answered_correctly_avg_c)+(1/train.answered_correctly_avg_u))
valid['hmean'] = 2 / ((1/valid.answered_correctly_avg_c)+(1/valid.answered_correctly_avg_u))

# use only last 30M training data for limited memory on kaggle env.
#train = train[-30000000:]

# part
questions_df = pd.read_csv(question_file)
train = pd.merge(train, questions_df[['question_id', 'part']], left_on = 'content_id', right_on = 'question_id', how = 'left')
valid = pd.merge(valid, questions_df[['question_id', 'part']], left_on = 'content_id', right_on = 'question_id', how = 'left')

# changing dtype to avoid lightgbm error
train['prior_question_had_explanation'] = train.prior_question_had_explanation.fillna(False).astype('int8')
valid['prior_question_had_explanation'] = valid.prior_question_had_explanation.fillna(False).astype('int8')


#MODELING
TARGET = 'answered_correctly'
FEATS = ['answered_correctly_avg_u', 'answered_correctly_sum_u', 'count_u', 'answered_correctly_avg_c', 'part', 'prior_question_elapsed_time', 'hmean', 'attempt']
dro_cols = list(set(train.columns) - set(FEATS))
y_tr = train[TARGET]
y_va = valid[TARGET]
train.drop(dro_cols, axis=1, inplace=True)
valid.drop(dro_cols, axis=1, inplace=True)
_=gc.collect()

lgb_train = lgb.Dataset(train[FEATS], y_tr)
lgb_valid = lgb.Dataset(valid[FEATS], y_va)
del train, y_tr
_=gc.collect()

model = lgb.train(
                    {'objective': 'binary',
                    'metric': 'auc',
                    'seed' : 11}, 
                    lgb_train,
                    valid_sets=[lgb_train, lgb_valid],
                    verbose_eval=100,
                    num_boost_round=10000,
                    early_stopping_rounds=10
                )
print('auc:', roc_auc_score(y_va, model.predict(valid[FEATS])))
_ = lgb.plot_importance(model)

import pickle
pickle.dump(model, open("lgbmodel_cv1.pickle", 'wb'))