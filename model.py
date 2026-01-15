import torch
import torch.nn as nn
from tqdm import tqdm

def _l2_normalize(x, dim=-1, eps=1e-12):
    return x / (x.norm(p=2, dim=dim, keepdim=True).clamp(min=eps))        

def _mean_pooling(last_hidden_state, attention_mask):
    # last_hidden_state: [B, L, H]; attention_mask: [B, L]
    mask = attention_mask.unsqueeze(-1)  # [B, L, 1]
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)  # [B, H]
    lens = mask.sum(dim=1).clamp(min=1)  # [B, 1]
    return summed / lens  

# class Cluster(object):
#     def __init__(self, args,df_papers,df_train,df_contexts):
#         self.df_papers = df_papers
#         self.df_train = df_train
#         self.df_contexts = df_contexts
#         self.n_nodes = args.num_nodes
#         self.train_queries_embs = self.train_queries_embedding(self.df_train,self.df_contexts) #[num_train,d]
#     def train_queries_embedding(self, df_train,df_contexts):
#         query_ids = df_train['context_id'].tolist()
#         texts = df_contexts.loc[query_ids]['masked_text'].fillna("").astype(str).tolist()
#         train_queries_embs = self._embed_texts(texts)  # 已 L2-normalized
#         return train_queries_embs  # [n_train, H] on GPU
        
#     def _embed_texts(self, texts,bar=True):
#         """对任意文本列表进行批量编码 -> CPU float32, L2-normalized"""
#         embs = []
#         with torch.no_grad():
#             for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding",disable=not bar):
#                 batch = texts[i:i + self.batch_size]
#                 enc = self.bert_tokenizer(
#                     batch,
#                     padding=True,
#                     truncation=True,
#                     max_length=self.max_length,
#                     return_tensors="pt"
#                 )
#                 enc = {k: v.to(self.device) for k, v in enc.items()}
#                 outputs = self.bert_model(**enc)
#                 if self.args.pooling_mode == 'mean':
#                     pooled = _mean_pooling(outputs.last_hidden_state, enc["attention_mask"])
#                 elif self.args.pooling_mode == 'cls':
#                     pooled = outputs.last_hidden_state[:, 0, :]
#                 pooled = _l2_normalize(pooled, dim=-1)
#                 embs.append(pooled)
#         return torch.cat(embs, dim=0).float().cuda()

class Graph_compression(nn.Module):
    def __init__(self, tokenizer,backbone,args, df_papers,df_train,df_contexts):
        super(Graph_compression, self).__init__()
        # common parameters
        self.args = args
        self.df_papers = df_papers
        self.df_train = df_train
        self.df_contexts = df_contexts
        self.batch_size = args.batch_size
        self.max_length = 512
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_tokenizer = tokenizer
        self.bert_model = backbone
        self.paper2id = {paper: i for i, paper in enumerate(df_papers.index.tolist())}
        self.id2paper = {i: paper for i, paper in enumerate(df_papers.index.tolist())}
        self.n_nodes = args.num_nodes
    def init_graph(self):
        self.train_queries_embs = self.train_queries_embedding(self.df_train,self.df_contexts) #[num_train,d]
        self.graph_embedding = nn.Parameter(self.train_queries_embs[:self.n_nodes]).to(self.device) # [n_nodes, H]
        # self.graph_embedding = nn.Parameter(self.kmeans_faiss(self.train_queries_embs, self.n_nodes)).to(self.device) # [n_nodes, H]
        # self.n_nodes = self.train_queries_embs.size(0)
        # self.graph_embedding = nn.Parameter(self.train_queries_embs).to(self.device) # [n_nodes, H]
    def train_queries_embedding(self, df_train,df_contexts):
        query_ids = df_train['context_id'].tolist()
        texts = df_contexts.loc[query_ids]['masked_text'].fillna("").astype(str).tolist()
        train_queries_embs = self._embed_texts(texts)  # 已 L2-normalized
        return train_queries_embs  # [n_train, H] on GPU
    
    def loss(self, query_ids, pos_ids):
        query_embs = self.train_queries_embs[query_ids] #[B, H]
        graph_embedding_norm = _l2_normalize(self.graph_embedding, dim=-1) #[n_nodes, H]
        sampled_routine = self.gumble_softmax(torch.matmul(query_embs, graph_embedding_norm.transpose(0,1)), temperature=1.0) #[B, n_nodes]
        query_graph_embs = torch.matmul(sampled_routine, graph_embedding_norm) #[B, H]
        # for semantic loss
        semantic_sims = torch.matmul(query_graph_embs, query_embs.transpose(0,1)) #[B, B]
        labels = torch.arange(semantic_sims.size(0), device=semantic_sims.device)
        loss1 = torch.nn.functional.cross_entropy(semantic_sims, labels)
        # for graph loss
        pos_embs = self.train_queries_embs[pos_ids] #[B, H]
        pos_routine = self.gumble_softmax(torch.matmul(pos_embs, graph_embedding_norm.transpose(0,1)), temperature=1.0) #[B, n_nodes]
        pos_graph_embs = torch.matmul(pos_routine, graph_embedding_norm) #[B, H]
        graph_sims = torch.matmul(query_graph_embs, pos_graph_embs.transpose(0,1)) #[B, B]
        loss2 = torch.nn.functional.cross_entropy(graph_sims, labels)
        #fusion loss
        loss = loss1 + loss2
        return loss1, loss2, loss
    def gumble_softmax(self, logits, temperature=0.01):
        # 1. 生成与 logits 同形状的 Gumbel 噪声
        # Gumbel(0, 1) 噪声可以通过两次取对数从均匀分布 U(0, 1) 中生成
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        # 2. 将噪声加到原始 logits 上，并应用温度参数
        y_soft = torch.softmax((logits + gumbel_noise) / temperature, dim=-1)
        return y_soft
    def graph(self):
        crow_indices = []
        col_indices = []
        values = []
        cluster = []
        with torch.no_grad():
            for i in tqdm(range(0, len(self.train_queries_embs), self.batch_size)):
                batch = self.train_queries_embs[i:i + self.batch_size] #[B, H]
                idx = torch.argmax(torch.matmul(batch, self.graph_embedding.transpose(0,1)), dim=1).tolist() #[B] 这个是把batch中每个query映射到graph中的哪个节点
                paper = self.df_contexts.loc[self.df_train['context_id'][i:i + self.batch_size]]['refid'].tolist()
                for j in range(len(idx)):
                    crow_indices.append(idx[j])
                    paper_ids = self.paper2id[paper[j]]
                    col_indices.append(paper_ids)
                    values.append(1)
                    cluster.append(idx[j])
        crow = torch.tensor(crow_indices,dtype=torch.int)
        col = torch.tensor(col_indices,dtype=torch.int)
        indices = torch.stack([crow, col], dim=0)  # [2, nnz]
        A = torch.sparse_coo_tensor(indices, torch.ones(len(crow_indices)), size=(self.n_nodes, len(self.df_papers)))
        A = A.coalesce()  # 自动合并重复 index 的值（默认是求和）
        return A , cluster#[n_nodes, num_paper]
                
    def _embed_texts(self, texts,bar=True):
        """对任意文本列表进行批量编码 -> CPU float32, L2-normalized"""
        embs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding",disable=not bar):
                batch = texts[i:i + self.batch_size]
                enc = self.bert_tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                outputs = self.bert_model(**enc)
                if self.args.pooling_mode == 'mean':
                    pooled = _mean_pooling(outputs.last_hidden_state, enc["attention_mask"])
                elif self.args.pooling_mode == 'cls':
                    pooled = outputs.last_hidden_state[:, 0, :]
                pooled = _l2_normalize(pooled, dim=-1)
                embs.append(pooled)
        return torch.cat(embs, dim=0)
    def query_alignment_loss(self, query, positive_query):
        q_enc = self.bert_tokenizer(query, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
        p_enc = self.bert_tokenizer(positive_query, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
        with torch.set_grad_enabled(True):
            q_emb = _mean_pooling(self.bert_model(**q_enc).last_hidden_state, q_enc['attention_mask']) #[B, H]
            p_emb = _mean_pooling(self.bert_model(**p_enc).last_hidden_state, p_enc['attention_mask']) #[B, H]
            q_emb = _l2_normalize(q_emb) #[B, H]
            p_emb = _l2_normalize(p_emb) #[B, H]
            # 相似度矩阵 BxB
            logits = torch.matmul(q_emb, p_emb.t())   # [B, B]
            labels = torch.arange(logits.size(0), device=logits.device)
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss.mean()
        
class Recommender(Graph_compression):
    def __init__(self, tokenizer,backbone,args, df_papers,df_train,df_contexts,A,graph_embedding):
        super(Recommender, self).__init__(tokenizer,backbone,args, df_papers,df_train,df_contexts)
        self.n_layers = 2
        self.n_neighbors = 5
        self.project = nn.Linear(768, 768).to(self.device)
        self.paper_embedding_rebuild()
        
        if self.args.mode == 'hybrid':
            self.graph_embedding = _l2_normalize(torch.tensor(graph_embedding).cuda()) #[num_paper,d]
            self.sp_context_paper = A.cuda() #[num_train,num_paper] Sparse matrix A
            self.sp_context_context =  torch.sparse.mm(self.sp_context_paper, self.sp_context_paper.transpose(0,1)).to_sparse() #[num_train,num_train] Sparse matrix S
            self.fusion_weight = nn.Parameter(torch.tensor(0.5,requires_grad=True))
            self.S = self.normalize_sparse_matrix(self.sp_context_context)
        if self.args.mode == 'flat':
            self.context_embedding = self._embed_texts(self.df_contexts.loc[self.df_train['context_id']]['masked_text'].fillna("").astype(str).tolist()) #[num_train,d]
            self.context_paper_adj = self.construct_context_paper_adj(self.df_train,self.df_contexts,self.df_papers).to_dense() #[num_train,num_paper]
    def construct_context_paper_adj(self,df_train,df_contexts,df_papers):
        crow_indices = []
        col_indices = []
        for context_id,context in enumerate(df_train['context_id'].tolist()):
            paper_id = self.paper2id[df_contexts.loc[context]['refid']]
            crow_indices.append(context_id)
            col_indices.append(paper_id)
        crow = torch.tensor(crow_indices,dtype=torch.int)
        col = torch.tensor(col_indices,dtype=torch.int)
        indices = torch.stack([crow, col], dim=0)  # [2, nnz]
        A = torch.sparse_coo_tensor(indices, torch.ones(len(crow_indices)), size=(len(self.df_train), len(self.paper2id)))
        return A.cuda()
    def paper_embedding_rebuild(self):
        self.paper_all_embs = self.paper_embedding(self.df_papers)  # [N, d], float32, on CPU, L2-normalized
    def normalize_sparse_matrix(self, A):
        indices = A.indices()
        r, c = indices
        deg = torch.sparse.sum(A, dim=1).to_dense()  # (N,)
        deg_inv_sqrt = torch.pow(deg + 1e-12, -0.5)
        norm_vals = A.values() * deg_inv_sqrt[r] * deg_inv_sqrt[c]
        S = torch.sparse_coo_tensor(indices, norm_vals, size=(A.size(0), A.size(1)))
        return S.coalesce()
    def propagate(self):
        E = self.project(self.graph_embedding)
        embs = [E]
        for _ in range(self.n_layers):
            E = torch.sparse.mm(self.S, E)
            embs.append(E)
        # layer-wise average
        out = torch.stack(embs, dim=0).mean(dim=0)
        return out #[num_train,768]
    def paper_embedding(self, df_papers):
        texts = df_papers["title_abstract"].fillna("").astype(str).tolist()
        paper_all_embs = self._embed_texts(texts)  # 已 L2-normalized
        return paper_all_embs  # [N, H] on GPU

    def merge_strategy(self, topk_idx_local1, topk_idx_local2, k=50, k2=1000):
        """
        合并两个推荐索引列表：
        - 保留 topk_idx_local1 的前 k 个；
        - 从 topk_idx_local2 中补充 k2 个去重的；
        - 输出形状为 [B, k + k2]。
        """
        B, K = topk_idx_local1.shape
        assert 0 <= k <= K, "k 必须在 [0, K] 范围内"

        merged_batches = []
        for b in range(B):
            keep = topk_idx_local1[b, :k]
            # 去重：排除 topk_idx_local1 中已有的项
            mask = ~torch.isin(topk_idx_local2[b], keep)
            rest = topk_idx_local2[b][mask][:k2]  # 取前 k2 个不重复项

            merged = torch.cat([keep, rest])
            merged_batches.append(merged)

        return torch.stack(merged_batches).tolist()  # [B, k + k2]

    def predict(self, inputx, query, cand_lists):
        """
        输入:
          - mode in {'query','title','abstract'}（保持你的接口）
          - cand_lists: List[List[int]] 大小约为 [batch_size, K]（候选在 self.all_embs 里的索引）
        输出:
          - topk_idx: List[List[int]] 形状 [batch_size, top_k]（在 cand_lists 局部排序后的索引）
        """
        # user_emb: [B, H]; 
        user_emb = self._embed_texts(inputx,bar=False)  # [B, H] on GPU
        query_emb = self._embed_texts(query,bar=False)  # [B, H] on GPU
        
        
        # 3) 取出候选向量并计算余弦相似度（因为都已归一化 -> 直接点积）
        #    先把 cand_lists 转成张量，做稳健的批量索引
        with torch.no_grad():
            if self.args.phase == 'prefetch':
                cands_embs = self.paper_all_embs #[K, H]
                scores1 = torch.matmul( user_emb, cands_embs.transpose(0,1))  # [B,K]
            else:
                cand_lists_tensor = torch.tensor(cand_lists, dtype=torch.int64).cuda()  # [B, K]
                cands_embs = self.paper_all_embs[cand_lists_tensor] #[B, K, H]
                scores1 = torch.sum( user_emb.unsqueeze(1) * cands_embs, dim=-1)  # [B, K]
                #score 2
            
            if self.args.mode == 'hybrid':
                sims = torch.matmul(query_emb, self.graph_embedding.t())  # [B, n_train] 
                q_ids = torch.topk(sims, k=self.n_neighbors, dim=1, largest=True).indices  # [B, 5]
                scores2 = self.q2q_score(q_ids) #[B, num_paper]
                if self.args.phase == 'prefetch':
                    scores2 = scores2 #[B, K]
                else:
                    scores2 = torch.gather(scores2, dim=1, index=cand_lists_tensor) #[B, K]
                score =  self.fusion_logits(scores1, scores2) 
            elif self.args.mode == 'vanilla':
                score = scores1
            else:
                topk_idx_sims = torch.matmul(query_emb, self.context_embedding.transpose(0,1)) #[B, num_train]
                index = torch.topk(topk_idx_sims, k=self.n_neighbors, dim=1, largest=True).indices #[B, 5]
                scores3 = self.context_paper_adj[index] #[B, n_neighbors, num_paper]
                scores3 = torch.sum(scores3, dim=1) #[B, num_paper]
                score = scores1 + scores3
            # 4) Top-K
            topk_idx_local1 = torch.argsort(score, dim=-1, descending=True)  # [B, top_k]
            topk_idx_local2 = torch.argsort(scores1, dim=-1, descending=True)  # [B, top_k]
            # 返回局部（在 cand_lists 内部）的排名索引（与原逻辑一致）
            # merge_num = 3 if self.args.phase == 'prefetch' else 5 # sota version
            topk_idx = self.merge_strategy(topk_idx_local1, topk_idx_local2, self.args.merge_num)
            candidate_ids = cand_lists.tolist()      # shape [B, N]
            candidate_reranked_ids = []
            for i in range(len(candidate_ids)):
                row_ids = []
                for j in range(len(topk_idx[i])):
                    row_ids.append(candidate_ids[i][topk_idx[i][j]])
                candidate_reranked_ids.append(row_ids)
            return candidate_reranked_ids

    def q2q_score(self, query_ids):
        if self.args.GCN:
            query_emb_GCN = self.propagate()
        else:
            query_emb_GCN = self.project(self.graph_embedding)
        batch_query_emb_GCN = query_emb_GCN[query_ids].mean(dim=1) #[B, H]
        sims = torch.matmul(
            batch_query_emb_GCN,  # [B, H]
            query_emb_GCN.transpose(0,1)  # [H, n_train]
        )  # [B, n_train]
        scores2 = torch.sparse.mm(sims,self.sp_context_paper) #[B, num_paper]
        scores2 = torch.sigmoid(scores2)
        return scores2
    def nce_loss(self, inputx, query, positive_doc, positive_doc_ids):  # q: B, p: B
        # inputs: query: B, positive_doc: B, positive_doc_ids: B
        # 编码
        q_enc = self.bert_tokenizer(inputx, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
        p_enc = self.bert_tokenizer(positive_doc, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
        query_enc = self.bert_tokenizer(query, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
        with torch.set_grad_enabled(True):
            q_emb = _mean_pooling(self.bert_model(**q_enc).last_hidden_state, q_enc['attention_mask'])
            p_emb = _mean_pooling(self.bert_model(**p_enc).last_hidden_state, p_enc['attention_mask'])
            query_emb = _mean_pooling(self.bert_model(**query_enc).last_hidden_state, query_enc['attention_mask'])
            # L2 归一化
            q_emb = _l2_normalize(q_emb) #[B, H]
            p_emb = _l2_normalize(p_emb) #[B, H]
            # 相似度矩阵 BxB
            logits1 = torch.matmul(q_emb, p_emb.t())   # [B, B]
            if self.args.mode == 'hybrid':
            # for query to query
                sims = torch.matmul(query_emb, self.graph_embedding.t())  # [B, n_train] 
                q_ids = torch.topk(sims, k=self.n_neighbors + 1, dim=1, largest=True).indices[:,1:]  # [B, 5]
                scores2 = self.q2q_score(q_ids) #[B, num_paper]
                logits2 = scores2[:,positive_doc_ids] #[B, B]
                logits = self.fusion_logits(logits1, logits2) #[B, B]
            elif self.args.mode in ['vanilla', 'flat']:
                logits = logits1 #[B, B]
            labels = torch.arange(logits.size(0), device=logits.device)
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss.mean()
    def fusion_logits(self, logits1, logits2):
        fusion_weight_clip = torch.clamp(self.fusion_weight, min=0, max=1)
        return (logits1*fusion_weight_clip + logits2 * (1-fusion_weight_clip))/0.05 #[B, B]