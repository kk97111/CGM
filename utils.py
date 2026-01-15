import json
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import random
import pandas as pd

import pandas as pd
def read_context(path):
    context_id = []
    citing_id = []
    refid = []
    masked_text = []
    context_json = json.load(open(path))
    for key in context_json.keys():
        context_id.append(str(key))
        citing_id.append(str(context_json[key]['citing_id']))
        refid.append(str(context_json[key]['refid']))
        masked_text.append(context_json[key]['masked_text'])
    df_contexts = pd.DataFrame({'context_id': context_id, 'citing_id':citing_id,'refid': refid, 'masked_text': masked_text})
    df_contexts.set_index('context_id',inplace=True)
    return df_contexts

def read_papers(path):
    paper_id = []
    title = []
    abstract = []
    title_abstract = []
    paper_json = json.load(open(path))
    for key in paper_json.keys():
        paper_id.append(str(key))
        title.append(str(paper_json[key]['title']))
        abstract.append(str(paper_json[key]['abstract']))
        title_abstract.append(str(paper_json[key]['title']) + " " + str(paper_json[key]['abstract']))
    df_papers = pd.DataFrame({'paper_id': paper_id, 'title': title, 'abstract': abstract, 'title_abstract': title_abstract})
    df_papers.set_index('paper_id',inplace=True)
    return df_papers
def read_train(path):
    context_id = []
    positive_ids = []
    train_json = json.load(open(path))
    for line in train_json:
        context_id.append(str(line['context_id']))
        positive_ids.append(line['positive_ids'])
    df_train = pd.DataFrame({'context_id': context_id, 'positive_ids': positive_ids})
    return df_train

def read_test(path):
    context_id = []
    positive_ids = []
    prefetched_ids = []
    candidate_name = 'candidates' if 'candidates' in path else 'prefetched_ids'
    with open(path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    data = data[0] if len(data) == 1 else data
    for line in data:
        context_id.append(str(line['context_id']))
        positive_ids.append(line['positive_ids'])
        prefetched_ids.append(line[candidate_name])
    df = pd.DataFrame({'context_id': context_id, 'positive_ids': positive_ids, 'candidates': prefetched_ids})
    df = reorder_candidates(df,100)
    return df

def reorder_candidates(df, top_k=100):
    """
    将 positive_ids 插入到 candidates 开头，去除重复项，并保留前 top_k 个。
    
    参数：
        df: pandas.DataFrame，必须包含 'positive_ids' 和 'candidates' 两列
        top_k: int，保留的候选数量（默认100）
    
    返回：
        pandas.DataFrame 处理后的 DataFrame
    """
    df = df.copy()  # 避免修改原始 df
    def adjust_candidates(row):
        pos_ids = row["positive_ids"]
        cands = row["candidates"]
        # 去除 positive_ids 中已有的
        cands = [c for c in cands if c not in pos_ids]
        # 插到最前面
        new_cands = pos_ids + cands
        # 保留前 top_k 个
        new_cands = new_cands[:top_k]
        random.shuffle(new_cands)
        return new_cands

    df["candidates"] = df.apply(adjust_candidates, axis=1)
    return df


# def load_raw_data(PATH_TO_DATA, dataset_name='peerread',splits = ['train', 'val', 'test', 'contexts', 'papers']):
#     data_dict = {}
#     for split in splits:
#         path_to_split = f"{PATH_TO_DATA}/{dataset_name}/{split}.json"
#         if 'train' in split or 'val' in split or 'test' in split:
#             dtype = {'context_id':str}
#         else:
#             dtype = None
#         try:
#             with open(path_to_split, "r", encoding="utf-8") as f:
#                 data_split = pd.read_json(f,dtype=dtype)
#         except:
#             with open(path_to_split, "r", encoding="utf-8") as f:
#                 data_split = pd.read_json(f,lines=True,dtype=dtype)
#         data_dict[split.split('_')[0]] = data_split
#     return data_dict



def text_merge(text1,text2):
    return [t1 + " " + t2 for t1, t2 in zip(text1, text2)]
    
def load_data(args,generate_candidates=False):
    PATH_TO_DATA = args.dataset_root
    dataset_name = args.dataset_name
    #load train data
    path_to_train = f"{PATH_TO_DATA}/{dataset_name}/train.json"
    df_train = read_train(path_to_train)
    #load test data
    if args.phase == 'prefetch':
        path_to_test = f"{PATH_TO_DATA}/{dataset_name}/test_with_oracle_prefetched_ids_for_reranking.json"
        df_test = read_test(path_to_test)
        df_test = df_test.drop(columns=['candidates'], errors='ignore')
    else:
        path_to_test = f"{PATH_TO_DATA}/{dataset_name}/test_candidates.json"
        df_test = read_test(path_to_test)
        df_test = df_test[df_test['candidates'].apply(len) >= 100]


    #load contexts data
    path_to_contexts = f"{PATH_TO_DATA}/{dataset_name}/contexts.json"
    df_contexts = read_context(path_to_contexts)
    #load papers data
    path_to_papers = f"{PATH_TO_DATA}/{dataset_name}/papers.json"
    df_papers = read_papers(path_to_papers)
    
    #对于超大数据集进行filter
    if dataset_name in ['refseer','arxiv']:
    # if dataset_name in ['refseer','arxiv','peerread','acl']:
        
    #     if dataset_name in  ['refseer','arxiv']:
    #         min_count = 5
    #     elif dataset_name in ['peerread','acl']:
    #         min_count = -1
        #去除低频引用文献
        refid_counts = df_contexts['refid'].value_counts()
        paper_corpus = set(refid_counts[refid_counts>5].index.tolist())
        
        df_contexts = df_contexts[df_contexts['refid'].isin(paper_corpus)]
        df_train = df_train[df_train['context_id'].isin(df_contexts.index.tolist())]
        df_test = df_test[df_test['context_id'].isin(df_contexts.index.tolist())]
        #去除低频上下文
        context_corpus = df_train['context_id'].tolist() + df_test['context_id'].tolist()
        df_contexts = df_contexts[df_contexts.index.isin(context_corpus)]
        paper_corpus = set(df_contexts['citing_id'].tolist() + df_contexts['refid'].tolist())
        df_papers = df_papers[df_papers.index.isin(paper_corpus)]
    
    print(f"Loaded {dataset_name} data: {len(df_train)} train, {len(df_test)} test, {len(df_contexts)} contexts, {len(df_papers)} papers")
    return {'train': df_train, 'test': df_test,'contexts': df_contexts,'papers': df_papers}

class CitationDataset(Dataset):
    def __init__(self, df_train,df_papers,df_contexts,negative_doc=None):
        self.df_papers = df_papers
        self.df_split = df_train
        self.paper2id = {paper:i for i,paper in enumerate(df_papers.index.tolist())}
        self.id2paper = {i:paper for paper,i in self.paper2id.items()}
        self.context_ids = df_train['context_id'].tolist() 
        self.context2id = {context:i for i,context in enumerate(self.context_ids)}
        self.queries = df_contexts.loc[self.context_ids]['masked_text'].tolist()
        self.citing_ids = df_contexts.loc[self.context_ids]['citing_id'].tolist()
        self.citing_titles = df_papers.loc[self.citing_ids]['title'].tolist()
        self.citing_abstracts = df_papers.loc[self.citing_ids]['abstract'].tolist()
        self.cited_ids = df_contexts.loc[self.context_ids]['refid'].tolist()
        self.cited_titles = df_papers.loc[self.cited_ids]['title'].tolist()
        self.cited_abstracts = df_papers.loc[self.cited_ids]['abstract'].tolist()
        self.candidates = df_train['candidates'].tolist()  if 'candidates' in df_train.columns else None   
        self.num_neg_samples = 5
        self.negative_doc = negative_doc
        self.paper2context = self.paper2context_mapping(df_contexts)
    def paper2context_mapping(self,df_contexts):
        paper2context = {i:[] for i in self.paper2id.keys()}
        for context in self.context_ids:
            paper_id = df_contexts.loc[context]['refid']
            paper2context[paper_id].append(context)
        return paper2context

    def __len__(self):
        return len(self.citing_ids)

    def __getitem__(self, idx):
        ret = {
            'citing_id': self.paper2id[self.citing_ids[idx]],
            'cited_id': self.paper2id[self.cited_ids[idx]],
            'query': self.queries[idx],
            'citing_title': self.citing_titles[idx],
            'citing_abstract': self.citing_abstracts[idx],
            'cited_title': self.cited_titles[idx],
            'cited_abstract': self.cited_abstracts[idx],
        }
        if self.candidates is None: #training phase
            ret['context_id'] = self.context2id[self.context_ids[idx]]
            c = [self.context2id[context] for context in self.paper2context[self.cited_ids[idx]]]
            ret['co_context_id'] = np.random.choice(c, 1)[0]
            ret['co_context'] = self.queries[ret['co_context_id']]
        if self.candidates is not None: #testing phase
            # 关键：变成 1D Tensor，防止 (1000,1) 这种列向量形状
            x = list(map(lambda x:self.paper2id[x],self.candidates[idx]))
            x = torch.tensor(x, dtype=torch.long).view(-1)  # => shape (1000,)
            ret['candidates'] = x
        return ret
