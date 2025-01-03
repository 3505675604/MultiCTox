import os
import pandas as pd
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader, Subset
from dgllife.utils import EarlyStopping
from nn_utils import NoamLR
from dataset import  collate,dataprocess
from utils import set_random_seed, evaluate
from model import Model
import config
import warnings
import pickle
warnings.filterwarnings("ignore")
from torch.optim import Adam
import numpy  as np
from torch.utils.data import BatchSampler, RandomSampler, DataLoader



def train_model(args, model, optimizer, scheduler, train_loader,device):
    total_all_loss = 0
    total_lab_loss = 0
    total_cl_loss = 0

    for batch1 in tqdm(train_loader): 
        batch1 = batch1.to(args.device)  
        labels = batch1.labels  
        model.zero_grad()

        out, ck = model(batch1.padded_smiles_batch, batch1, batch1.fps_t)

        zs = torch.sigmoid(out)
        preds = (zs >= 0.5).float()  

        ts = labels.float()

        all_loss, lab_loss, cl_loss = model.loss_cal(ck, preds, ts)

        total_all_loss += all_loss.item()
        total_lab_loss += lab_loss.item()
        total_cl_loss += cl_loss.item()

        all_loss.backward()
        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()

    return total_all_loss, total_lab_loss, total_cl_loss, model

def train(args, train_loader, val_loader, model, loss_func, optimizer, scheduler, stopper):
    batch2=[]
    for batch1 in train_loader:
        batch2.append(batch1.to(args.device))


    batch3 = []

    for batch4 in val_loader:
        batch3.append(batch4.to(args.device))

    for epoch in range(args.epoch):
        model.train()
        # one_batch_bar = tqdm(train_loader, ncols=100)
        # one_batch_bar.set_description(f'[iter:{args.iter},epoch:{epoch + 1}/{args.epoch}]')
        cur_lr = optimizer.param_groups[0]["lr"]
        res1=[]
        for i, batch1 in enumerate(batch2):

            labels = batch1.labels


            pred,loss1 = model(batch1.padded_smiles_batch, batch1, batch1.fps_t)
            AC, F1, SN, SP, CCR, MCC = evaluate(labels.unsqueeze(1), pred)
            res1.append([AC, F1, SN, SP, CCR, MCC])

            # acc, precision, recall, f1score, acc_weight = evaluate(labels, pred)
            loss = loss_func(pred, labels.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_results = pd.DataFrame(res1, columns=['AC', 'F1', 'SN', 'SP', 'CCR', 'MCC'])
        r1 = train_results.mean()


        print(
            f"epoch:{epoch}---train---AC:{r1['AC']}---F1:{r1['F1']}---SN:{r1['SN']}---SP:{r1['SP']}---CCR:{r1['CCR']}---MCC:{r1['MCC']}-"
                 )

        scheduler.step()
        model.eval()
        res = []

        with torch.no_grad():
            for batch in batch3:
                # batch = batch.to(args.device)
                labels = batch.labels
                padded_smiles_batch = batch.padded_smiles_batch

                fps_t = batch.fps_t
                pred,loss1 = model(padded_smiles_batch, batch, fps_t)
                AC, F1, SN, SP, CCR,MCC = evaluate(labels.unsqueeze(1), pred)
                res.append([AC, F1, SN, SP,CCR,MCC])
        val_results = pd.DataFrame(res, columns=['AC', 'F1', 'SN', 'SP','CCR','MCC'])
        r = val_results.mean()
        print(
            f"epoch:{epoch}---validation---AC:{r['AC']}---F1:{r['F1']}---SN:{r['SN']}---SP:{r['SP']}---CCR:{r['CCR']}---MCC:{r['MCC']}-"
            f"---lr:{cur_lr}")
        early_stop = stopper.step(r['AC'], model)
        if early_stop:
            break


def main(args):
    source = 'hERG'  # Cav1.2/hERG/Nav1.5
    datas = 'herg'  # cav/herg/nav
    dataspathmulu='/root/autodl-tmp/MultiCBlo-master/data/' + source + '/'
    data_path = '/root/autodl-tmp/MultiCBlo-master/data/' + source + '/data_' + datas + '_dev.csv'
    data_path_test = '/root/autodl-tmp/MultiCBlo-master/data/' + source + '/eval_set_' + datas + '_60.csv'
    data_path_test1 = '/root/autodl-tmp/MultiCBlo-master/data/' + source + '/eval_set_' + datas + '_70.csv'
    data_index = pd.read_csv('/root/autodl-tmp/MultiCBlo-master/data/' + source + '/data_' + datas + '_dev.csv')



    # if os.path.exists(dataspathmulu+'dataset.pkl'):
    #     with open(dataspathmulu+'dataset.pkl', 'rb') as file:
    #         dataset = pickle.load(file)
    #     with open(dataspathmulu+'test_dataset.pkl', 'rb') as file:
    #         test_dataset = pickle.load(file)
    #     with open(dataspathmulu+'test_dataset1.pkl', 'rb') as file:
    #         test_dataset1 = pickle.load(file)
    # else:
    dataset = dataprocess(data_path,dataspathmulu+'dataset.pkl')
    test_dataset = dataprocess(data_path_test,dataspathmulu+'test_dataset.pkl')
    test_dataset1 = dataprocess(data_path_test1,dataspathmulu+'test_dataset1.pkl')
        



    data_index_train=data_index.index[data_index['USED_AS'] == 'Train'].tolist()
    data_index_Validation=data_index.index[data_index['USED_AS'] == 'Validation'].tolist()


    train_dataset = Subset(dataset, data_index_train)
    validate_dataset = Subset(dataset,data_index_Validation)
    n_feats = 84

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate)
    val_loader = DataLoader(validate_dataset, batch_size=args.batch_size, collate_fn=collate)
    test_loader60 = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate)
    test_loader70 = DataLoader(test_dataset1, batch_size=args.batch_size, collate_fn=collate)
    mean_results = []
    mean_results1 = []


#--------------------------------------------------------------------------------------------------------------------------------------------------------------


    #train model 
    # data_name='esol'
    # dataspathmulu1='/root/autodl-tmp/MultiCBlo-master/data/train_mdoel_data/'
    # data_train_model_path='/root/autodl-tmp/MultiCBlo-master/data/train_mdoel_data/'+data_name+'.csv'
    
    # data_train_model_smiles=pd.read_csv(data_train_model_path)

    # if os.path.exists(dataspathmulu+'esol.pkl'):
    #     with open(dataspathmulu+'esol.pkl', 'rb') as file:
    #         test_dataset1 = pickle.load(file)
    # else:
    #     data_train_model_embeddings=dataprocess(data_train_model_path,dataspathmulu1+'esol.pkl')

    # data_train_model_idx = data_train_model_smiles.index.tolist()

    # ids = list(range(len(data_train_model_idx)))
    # train_sampler = RandomSampler(data_train_model_idx)
    # train_model_idx_loader = DataLoader(data_train_model_embeddings, batch_size=args.batch_size, sampler=train_sampler,collate_fn=collate)


    # model = Model( in_feats=n_feats, hidden_feats=args.hidden_feats,
    #             rnn_embed_dim=args.rnn_embed_dim, blstm_dim=args.rnn_hidden_dim, blstm_layers=args.rnn_layers,
    #             fp_2_dim=args.fp_dim, dropout=args.p, num_heads=args.head, device=args.device).to(args.device)
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = Adam(params=model.parameters(), lr=1e-3, weight_decay=1e-5)
    # schedule = NoamLR(optimizer=optimizer, warmup_epochs=[2.0], total_epochs=[2000],
    #                   steps_per_epoch=len(data_train_model_idx) // args.batch_size, init_lr=[1e-3],
    #                   max_lr=[2e-3], final_lr=[1e-3], )

    # for epoch in range(1000):
    #     np.random.shuffle(ids)
    #     train_all_loss, train_lab_loss, train_cl_loss, model_after_train = train_model(args, model, optimizer, schedule, train_model_idx_loader)

    #     print('train model loss',train_all_loss, train_lab_loss, train_cl_loss)
    #     torch.cuda.empty_cache()


    # data_name = 'esol'#esol,freesolv
    # dataspathmulu1 = '/root/autodl-tmp/MultiCBlo-master/data/train_mdoel_data/'
    # data_train_model_path = dataspathmulu1 + data_name + '.csv'

    # data_train_model_smiles = pd.read_csv(data_train_model_path)

    # if os.path.exists(dataspathmulu1 + data_name+'.pkl'):
    #     with open(dataspathmulu1 + data_name+'.pkl', 'rb') as file:
    #         data_train_model_embeddings = pickle.load(file)
    # else:
    #     data_train_model_embeddings = dataprocess(data_train_model_path, dataspathmulu1 + data_name+'.pkl')

    # data_train_model_idx = data_train_model_smiles.index.tolist()

    # ids = list(range(len(data_train_model_idx)))
    # train_sampler = RandomSampler(data_train_model_idx)
    # train_model_idx_loader = DataLoader(data_train_model_embeddings, batch_size=args.batch_size1, sampler=train_sampler, collate_fn=collate)

    # model = Model(
    #     in_feats=n_feats, hidden_feats=args.hidden_feats,
    #     rnn_embed_dim=args.rnn_embed_dim, blstm_dim=args.rnn_hidden_dim, blstm_layers=args.rnn_layers,
    #     fp_2_dim=args.fp_dim, dropout=args.p, num_heads=args.head, device=args.device
    # ).to(args.device)

    # optimizer = Adam(params=model.parameters(), lr=1e-3, weight_decay=1e-5)

    # # schedule = NoamLR(
    # #     optimizer=optimizer, warmup_epochs=[2.0], total_epochs=[2000],
    # #     steps_per_epoch=len(data_train_model_idx) // args.batch_size1, init_lr=[1e-3],
    # #     max_lr=[2e-3], final_lr=[1e-3]
    # # )
    # optimizer = Adam(params=model.parameters(), lr=args.init_lr, weight_decay=1e-5)
    # schedule = NoamLR(optimizer=optimizer, warmup_epochs=[args.warmup_epochs], total_epochs=[100],
    #                   steps_per_epoch=len(data_train_model_idx) // args.batch_size1, init_lr=[args.init_lr],
    #                   max_lr=[args.max_lr], final_lr=[args.final_lr], )

    # for epoch in range(1000):
    #     np.random.shuffle(ids)
        
    #     train_all_loss, train_lab_loss, train_cl_loss, model_after_train = train_model(
    #         args, model, optimizer, schedule, train_model_idx_loader, device=args.device
    #     )

    #     print(f'Epoch [{epoch + 1}/1000] | Total Loss: {train_all_loss:.4f} | Lab Loss: {train_lab_loss:.4f} | CL Loss: {train_cl_loss:.4f}')

    #     torch.cuda.empty_cache()


#--------------------------------------------------------------------------------------------------------------------------------------------------------------

    # train

    for iteration in range(args.iterations):
        args.iter = iteration
        model = Model( in_feats=n_feats, hidden_feats=args.hidden_feats,
                    rnn_embed_dim=args.rnn_embed_dim, blstm_dim=args.rnn_hidden_dim, blstm_layers=args.rnn_layers,
                    fp_2_dim=args.fp_dim, dropout=args.p, num_heads=args.head, device=args.device).to(args.device)
        # model=model_after_train
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
        stopper = EarlyStopping(mode='higher', filename=f'{args.output}/net_{iteration}.pkl', patience= 50)
        loss_func = torch.nn.BCEWithLogitsLoss()
        train(args, train_loader, val_loader, model, loss_func, optimizer, scheduler, stopper)
        stopper.load_checkpoint(model)
        model.eval()
        total_preds = torch.Tensor()
        total_labels = torch.Tensor()
        total_preds1 = torch.Tensor()
        total_labels1 = torch.Tensor()
        res60 = []
        res70 = []
        with torch.no_grad():
            for batch in test_loader60:
                batch = batch.to(args.device)
                labels = batch.labels
                padded_smiles_batch = batch.padded_smiles_batch
                fps_t = batch.fps_t
                pred,loss1= model(padded_smiles_batch, batch, fps_t)
                total_preds = torch.cat((total_preds, pred.cpu()), 0)
                total_labels = torch.cat((total_labels, labels.cpu()), 0)

            AC, F1, SN, SP, CCR,MCC  = evaluate(total_labels.unsqueeze(1), total_preds)
            res60.append([AC, F1, SN, SP, CCR,MCC])
        # with torch.no_grad():

            for batch in test_loader70:
                batch = batch.to(args.device)
                labels = batch.labels
                padded_smiles_batch = batch.padded_smiles_batch

                fps_t = batch.fps_t
                pred,loss1= model(padded_smiles_batch, batch, fps_t)
                total_preds1 = torch.cat((total_preds1, pred.cpu()), 0)
                total_labels1 = torch.cat((total_labels1, labels.cpu()), 0)

            AC, F1, SN, SP, CCR, MCC = evaluate(total_labels1.unsqueeze(1), total_preds1)
            res70.append([AC, F1, SN, SP, CCR, MCC])

        test_results60 = pd.DataFrame(res60, columns=['AC', 'F1', 'SN', 'SP','CCR','MCC'])
        r = test_results60.mean()
        print("60:")
        print(f"test_---AC:{r['AC']}---F1:{r['F1']}---SN:{r['SN']}---SP:{r['SP']}---CCR:{r['CCR']}---MCC:{r['MCC']}-")
        mean_results.append([r['AC'], r['F1'], r['SN'], r['SP'], r['CCR'],r['MCC']])
        test_mean_results = pd.DataFrame(mean_results, columns=['AC', 'F1', 'SN', 'SP','CCR','MCC'])
        r = test_mean_results.mean()
        print(
            f"mean_test_---AC:{r['AC']}---F1:{r['F1']}---SN:{r['SN']}---SP:{r['SP']}---CCR:{r['CCR']}---MCC:{r['MCC']}-")
        test_mean_results.to_csv(f'{args.output}/10_test_results60.csv', index=False)

        test_results70 = pd.DataFrame(res70, columns=['AC', 'F1', 'SN', 'SP','CCR','MCC'])
        r = test_results70.mean()
        print("70:")

        print(f"test_---AC:{r['AC']}---F1:{r['F1']}---SN:{r['SN']}---SP:{r['SP']}---CCR:{r['CCR']}---MCC:{r['MCC']}-")
        mean_results1.append([r['AC'], r['F1'], r['SN'], r['SP'], r['CCR'],r['MCC']])
        test_mean_results = pd.DataFrame(mean_results1, columns=['AC', 'F1', 'SN', 'SP','CCR','MCC'])
        r = test_mean_results.mean()
        print(
            f"mean_test_---AC:{r['AC']}---F1:{r['F1']}---SN:{r['SN']}---SP:{r['SP']}---CCR:{r['CCR']}---MCC:{r['MCC']}-")
        test_mean_results.to_csv(f'{args.output}/10_test_results70.csv', index=False)


if __name__ == '__main__':
    args = config.parse()
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_random_seed(args.seed)
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    main(args)
