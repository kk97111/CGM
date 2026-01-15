import numpy as np
import torch
from utils import load_data,CitationDataset,text_merge
from huggingface_hub import login
import argparse
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from baselines.evaluation import display_metrics
from model import Graph_compression, Recommender
from datetime import datetime
# import faiss
# login("your_huggingface_token")  




def evaluate(model,test_loader,df_contexts,paper_corpus):
    # Prediction
    model.paper_embedding_rebuild()
    results = []

    for batch in tqdm(test_loader):
        query = batch['query']
        inputx = text_merge(text_merge(batch['query'],batch['citing_title']),batch['citing_abstract'])
        if model.args.phase == 'rerank':
            candidates = batch['candidates']
        else:
            candidates = np.repeat(np.arange(len(paper_corpus)).reshape(1,-1), len(query), axis=0)#.tolist()
        result = model.predict(inputx,query,candidates)
        results.extend(result)
    # Evaluation
    df_test = test_loader.dataset.df_split
    GT_ids = list(map(lambda x:model.paper2id[x],df_contexts.loc[df_test['context_id']]['refid'].tolist()))
    if args.phase == 'rerank':
        k_list = (1,5,10,30,50,80,100)
    elif args.phase == 'prefetch':
        k_list = (1,5,10,25,50,100,250,500,1000)
    metrics_df = display_metrics(results, GT_ids, K_LIST=k_list)
    return metrics_df


def main(args):
    #读取数据
    print(f"Dataset: {args.dataset_name}, phase: {args.phase}, mode: {args.mode}, graph_node: {args.num_nodes}, GCN: {args.GCN}, context_alignment: {args.context_alignment}\n")
    data_dict = load_data(args)
    df_papers = data_dict['papers']
    df_contexts = data_dict['contexts']
    df_train = data_dict['train']
    df_test = data_dict['test']

    train_dataset = CitationDataset(df_train,df_papers,df_contexts)
    test_dataset = CitationDataset(df_test,df_papers,df_contexts)

    train_loader = DataLoader(train_dataset,
        batch_size=args.batch_size,
        shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=False)
    #load tokenizer and backbone
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    backbone = AutoModel.from_pretrained("allenai/specter2_base").cuda()    
    # for graph compression
    A, graph_embedding = None, None
    if args.mode == 'hybrid':
        graph_compression = Graph_compression(tokenizer,backbone,args,df_papers,df_train,df_contexts)
        graph_compression.init_graph()
        optimizer_graph = torch.optim.Adam(graph_compression.parameters(), lr=5e-5)
        # context alignment
        if args.context_alignment:
            n_epoch = 0 if args.dataset_name in ['refseer','arxiv'] else 0 #[这个地方我改了]
            for epoch in range(n_epoch):
                for j,batch in enumerate(tqdm(train_loader,desc='Graph alignment Epoch {epoch}')):
                    query = batch['query']
                    positive_query = batch['co_context']
                    loss = graph_compression.query_alignment_loss(query, positive_query)
                    loss.backward()
                    optimizer_graph.step()
                    optimizer_graph.zero_grad()
                    if j % 10000 == 10000-2 and args.dataset_name in ['refseer','arxiv']:
                        break
        # graph_compression.init_graph()
        loss1_list, loss2_list, loss_list = [], [], []
        optimizer_graph = torch.optim.Adam(graph_compression.parameters(), lr=0.001)
        n_epoch = 5 if args.dataset_name in ['refseer','arxiv'] else 50
        for epoch in range(n_epoch):
            for batch in tqdm(train_loader,desc=f'Graph Compression Epoch {epoch}'):
                query_ids = batch['context_id']
                pos_ids = batch['co_context_id']
                loss1, loss2, loss_both = graph_compression.loss(query_ids, pos_ids)
                if args.compression_mode == 'both':
                    loss = loss_both
                elif args.compression_mode == 'structure':
                    loss = loss1
                elif args.compression_mode == 'semantic':
                    loss = loss2
                loss1_list.append(loss1.item())
                loss2_list.append(loss2.item())
                loss_list.append(loss_both.item())
                loss.backward()
                optimizer_graph.step()
                optimizer_graph.zero_grad()   
            tqdm.write(f'Graph Compression Epoch {epoch}, Loss1: {np.mean(loss1_list)}, Loss2: {np.mean(loss2_list)}, Loss: {np.mean(loss_list)}')

        A, cluster = graph_compression.graph()
        graph_embedding = graph_compression.graph_embedding.detach().cpu().numpy()
        del graph_compression

    #train recommender
    model = Recommender(tokenizer,backbone,args,df_papers,df_train,df_contexts,A,graph_embedding)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    batch_total = 0
    best_metrics = - np.inf
    tolerance = 0
    flag = True
    start_time = datetime.now()
    for epoch in range(args.epochs):
        if not flag:
            break
        for batch in tqdm(train_loader,desc=f'Epoch {epoch}'):
            inputx = text_merge(text_merge(batch['query'],batch['citing_title']),batch['citing_abstract'])
            query = batch['query']
            positive_document = text_merge(batch['cited_title'],batch['cited_abstract'])
            positive_query = batch['co_context']
            # negative_document = text_merge(batch['negative_title'],batch['negative_abstract'])
            positive_document_ids = torch.tensor(batch['cited_id'],dtype=torch.int64).cuda() #[B]
            loss1 = model.nce_loss(inputx,query,positive_document,positive_document_ids)
            loss2 = model.query_alignment_loss(query, positive_query)
            loss = loss1 + 0.5 * loss2 if args.context_alignment else loss1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch_total += 1
            per_test = 50000 if args.dataset_name == 'refseer' else 2500 if args.dataset_name == 'arxiv' else 200
            if batch_total % per_test == 1:
                print(f"Evaluating at batch {batch_total}")
                metrics_df = evaluate(model,test_loader,df_contexts,df_papers.index.tolist())
                # print(metrics_df)
                if float(metrics_df['NDCG'][5]) > best_metrics:
                    best_metrics = float(metrics_df['NDCG'][5])
                    save_metrics = metrics_df
                    tolerance = 0
                    if args.dataset_name in ['refseer','arxiv']:
                        torch.save(model.state_dict(), f'./model/{args.dataset_name}_{args.mode}_model.pth')
                        np.save(f'./model/{args.dataset_name}_{args.mode}_graph_embedding.npy', graph_embedding)
                        torch.save({'indices': A.indices(), 'values': A.values(), 'size': A.size()}, f'./model/{args.dataset_name}_{args.mode}_A.pth')
                else:
                    tolerance += 1
                    if tolerance >= 3:
                        flag = False
        #     break
        # break
    end_time = datetime.now()
    metrics_df = evaluate(model,test_loader,df_contexts,df_papers.index.tolist())
    print(f"\n\nMetrics:")
    print(save_metrics.transpose())
    print('--------------------------------')
    os.makedirs("./results", exist_ok=True)
    file_name = f'./results/{args.dataset_name}_{args.phase}_{args.mode}_nodes{args.num_nodes}_merge{args.merge_num}_cmode{args.compression_mode}.txt'

    
    
    with open(file_name, 'a+') as f:
        text = f"\n\n{'='*50}{start_time.strftime('%Y-%m-%d %H:%M:%S')}{'='*50}\n"
        text += f"Dataset: {args.dataset_name}, phase: {args.phase}, mode: {args.mode}, graph_node: {args.num_nodes}, GCN: {args.GCN}, compression_mode: {args.compression_mode}, epochs: {args.epochs}, Merge Num: {args.merge_num}\n"
        text += f"Metrics: {save_metrics}\n"
        text += f"{'='*50}{end_time.strftime('%Y-%m-%d %H:%M:%S')}{'='*50}\n"
        text += f"Time taken: {(end_time - start_time)}\n"
        text += f"{'='*50}\n\n"
        f.write(text)
    
    print(f"Result saved to {file_name}")



    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='scibert')
    parser.add_argument('--dataset_root', type=str, default='your_path_to_dataset')
    parser.add_argument('--dataset_name', type=str, default='peerread')
    parser.add_argument('--top_k', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--pooling_mode', type=str, default='cls')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--num_nodes', type=int, default=5000)
    parser.add_argument('--phase', type=str, default='rerank') # rerank, prefetch
    parser.add_argument('--mode', type=str, default='vanilla') #hybrid, vanilla
    parser.add_argument('--compression_mode', type=str, default='both') #both, structure, semantic
    parser.add_argument('--GCN', type=bool, default=True, help='Disable GCN (default: enabled)')   
    parser.add_argument('--context_alignment', type=bool, default=False, help='Disable context alignment (default: enabled)')
    parser.add_argument('--merge_num', type=int, default=5, help='Merge number (default: 5)')

    args = parser.parse_args()  
    main(args)
