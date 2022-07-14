import json
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from cotk.metric import MetricChain, PerplexityMetric
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import BertTokenizer, BartForConditionalGeneration, get_linear_schedule_with_warmup
from transformers.file_utils import ModelOutput
from metric import BleuCorpusMetric, SingleTurnDistinct
from utils.HGT import HGT
from utils.utils import get_klg2id, process_load
from torch_geometric.data import HeteroData, Batch


class KdConvDataset(Dataset):
    def __init__(self, file_id, key, device_id, num_turns=8, num_workers=8):
        super().__init__()
        assert key in ['train', 'dev', 'test']
        self.device = torch.device("cuda:%d" % device_id if torch.cuda.is_available() else "cpu")
        self.num_turns = num_turns
        self.num_workers = num_workers

        self.tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
        self.entity2id, self.relation2id, self.triple2id, self.kg2dialkg = get_klg2id(file_id)
        self.triple_embed_all, self.relation_embed_all = process_load(file_id)
        print("%sData loading..." % key)
        self.datas = json.load(open('%s/%s.json' % (file_id, key), encoding='utf8'))
        self.origin_data = self.prepare_data(self.datas)
        print("Load complete: %d" % len(self.origin_data['posts']))

    def prepare_data(self, datas):
        origin_data = {'posts': [], 'responses': [], 't_post': [], 'r_post': [], 't2t': [], 'r2r': [], 'k_info': [], 't_gt': [], 'r_gt': []}
        k_train, r_train = [], []
        for data in datas:
            messages = data['messages']
            posts, resps = [], []
            # turn = []
            k_dial, r_dial = [], []
            k_info = []
            for message in messages:
                sent = message['message']
                posts.append(sent)
                k_sent, r_sent = [], []
                info_sent = ""
                if 'attrs' in message:
                    for attr in message['attrs']:
                        if attr['attrname'] == 'Information':
                            info_sent = attr['attrvalue']
                            continue
                        h = self.entity2id[attr['name'].replace('【', '').replace('】', '')]
                        r = self.relation2id[attr['attrname'].replace('【', '').replace('】', '')]
                        t = self.entity2id[attr['attrvalue'].replace('【', '').replace('】', '')]

                        if (h, r, t) in self.triple2id.keys():
                            k_sent.append(self.triple2id[(h, r, t)])
                            r_sent.append(r)
                    # if len(k_sent) != 0:
                    #     resps.append(sent)
                k_dial.append(k_sent)
                r_dial.append(r_sent)
                k_info.append(info_sent)
            k_train.append(k_dial)
            r_train.append(r_dial)

            for i in range(1, len(posts)):
                origin_data['posts'].append([posts[j] for j in range(max(0, (i + 1) - (self.num_turns - 1)), i)])
                origin_data['responses'].append(posts[i])
                origin_data['t_post'].append([])
                origin_data['r_post'].append([])
                origin_data['t2t'].append([[], []])
                origin_data['r2r'].append([[], []])
                origin_data['t_gt'].append([])
                origin_data['r_gt'].append([])
                origin_data['k_info'].append(k_info[i])
                for ki, k_sent in enumerate(k_dial[:i]):
                    if i < 2:
                        origin_data['t_post'][-1].extend(k_sent)
                        origin_data['r_post'][-1].extend(r_dial[ki])
                        break

                    if ki < i - 1:
                        # t2t_sent = [[], []]
                        # r2r_sent = [[], []]
                        for j1, t1 in enumerate(k_sent):
                            for j2, t2 in enumerate(k_dial[ki + 1]):
                                origin_data['t2t'][-1][0].append(t1)
                                origin_data['t2t'][-1][1].append(t2)
                                origin_data['r2r'][-1][0].append(r_dial[ki][j1])
                                origin_data['r2r'][-1][1].append(r_dial[ki + 1][j2])
                        # origin_data['t2t'][-1].extend(t2t_sent)
                        # origin_data['r2r'][-1].extend(r2r_sent)

                    origin_data['t_post'][-1].extend(k_sent)
                    origin_data['r_post'][-1].extend(r_dial[ki])

                origin_data['t_gt'][-1].extend(k_dial[i])
                origin_data['r_gt'][-1].extend(r_dial[i])

        return origin_data

    def __len__(self):
        return len(self.origin_data['posts'])

    def __getitem__(self, idx):
        posts_item = self.origin_data['posts'][idx]
        resp_item = self.origin_data['responses'][idx]
        posts_id = torch.tensor([self.tokenizer.cls_token_id]).to(self.device)
        for i, p in enumerate(posts_item):
            post_id = self.tokenizer(p, padding=False, truncation=True, return_tensors="pt").to(self.device)
            posts_id = torch.cat([posts_id, post_id.input_ids.squeeze(0)[1:]], dim=0)
            if posts_id.shape[0] > 510:
                # print("posts长度：", posts_id.shape)
                posts_id = posts_id[-510:]
        resp_id = self.tokenizer(resp_item, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        resp_id = resp_id.input_ids.squeeze(0)

        hetero_data = HeteroData()

        triple_order, relation_order = [], []
        tri2orderid, rel2orderid = {}, {}
        if len(self.origin_data['t_post'][idx]) == 0:
            hetero_data['triple'].x = torch.tensor(self.origin_data['t_post'][idx], dtype=torch.float)
            hetero_data['relation'].x = torch.tensor(self.origin_data['r_post'][idx], dtype=torch.float)
            hetero_data['triple', 'contains', 'relation'].edge_index = torch.tensor([self.origin_data['t_post'][idx], self.origin_data['r_post'][idx]], dtype=torch.long)
            hetero_data['relation', 'belongsto', 'triple'].edge_index = torch.tensor([self.origin_data['r_post'][idx], self.origin_data['t_post'][idx]], dtype=torch.long)
        else:
            for i, tr in enumerate(zip(self.origin_data['t_post'][idx], self.origin_data['r_post'][idx])):
                triple_order.append(self.triple_embed_all[tr[0]])
                relation_order.append(self.relation_embed_all[tr[1]])
                tri2orderid[tr[0]] = i
                rel2orderid[tr[1]] = i
            hetero_data['triple'].x = torch.tensor(triple_order, dtype=torch.float)
            hetero_data['relation'].x = torch.tensor(relation_order, dtype=torch.float)
            t2r = [[tri2orderid[t] for t in self.origin_data['t_post'][idx]], [rel2orderid[r] for r in self.origin_data['r_post'][idx]]]
            r2t = [[rel2orderid[r] for r in self.origin_data['r_post'][idx]], [tri2orderid[t] for t in self.origin_data['t_post'][idx]]]
            hetero_data['triple', 'contains', 'relation'].edge_index = torch.tensor(t2r, dtype=torch.long)
            hetero_data['relation', 'belongsto', 'triple'].edge_index = torch.tensor(r2t, dtype=torch.long)

        if len(self.origin_data['t2t'][idx][0]) != 0:
            t2t = [[tri2orderid[t] for t in self.origin_data['t2t'][idx][0]], [tri2orderid[t] for t in self.origin_data['t2t'][idx][1]]]
            r2r = [[rel2orderid[r] for r in self.origin_data['r2r'][idx][0]], [rel2orderid[r] for r in self.origin_data['r2r'][idx][1]]]
            hetero_data['triple', 'next_klg', 'triple'].edge_index = torch.tensor(t2t, dtype=torch.long)
            hetero_data['relation', 'next_rel', 'relation'].edge_index = torch.tensor(r2r, dtype=torch.long)
        else:
            hetero_data['triple', 'next_klg', 'triple'].edge_index = torch.tensor(self.origin_data['t2t'][idx], dtype=torch.long)
            hetero_data['relation', 'next_rel', 'relation'].edge_index = torch.tensor(self.origin_data['r2r'][idx], dtype=torch.long)

        # 求gt的平均值
        if len(self.origin_data['t_gt'][idx]) > 0:
            tgt = torch.tensor([self.triple_embed_all[t] for t in self.origin_data['t_gt'][idx]])
            tgt_mean = torch.mean(tgt, dim=0, keepdim=True)
        else:
            tgt_mean = torch.zeros([1, 600])
        # rgt = torch.tensor([self.relation_embed_all[r] for r in self.origin_data['r_gt'][idx]])
        # rgt_mean = torch.mean(rgt, dim=0, keepdim=True)

        # 转换为gt的id
        tgt_id = torch.zeros(len(self.triple_embed_all))
        for t in self.origin_data['t_gt'][idx]:
            tgt_id[t] = 1
        tgt_id = tgt_id.unsqueeze(0)
        rgt_id = torch.zeros(len(self.relation_embed_all))
        for r in self.origin_data['r_gt'][idx]:
            rgt_id[r] = 1
        rgt_id = rgt_id.unsqueeze(0)

        # 转换为dialkg的id
        # tgt_id = torch.zeros(len(self.kg2dialkg) + 1)
        # for t in self.origin_data['t_gt'][idx]:
        #     d = int(self.kg2dialkg[t])
        #     tgt_id[d] = 1
        # tgt_id = tgt_id.unsqueeze(0)

        info_ids = self.tokenizer(self.origin_data['k_info'][idx], padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        info_ids = info_ids.input_ids.squeeze(0)

        return posts_id, resp_id, hetero_data, tgt_id, tgt_mean, rgt_id, info_ids

    @staticmethod
    def collate_fn(batch):
        pad_token_id = 0
        post_ids, resp_ids, hetero_data, t_gt, tgt_mean, r_gt, info = list(zip(*batch))
        post_batchids = torch.nn.utils.rnn.pad_sequence(post_ids, batch_first=True, padding_value=pad_token_id)
        resp_batchids = torch.nn.utils.rnn.pad_sequence(resp_ids, batch_first=True, padding_value=pad_token_id)
        t_gt_batch = torch.cat(t_gt, dim=0)
        t_gt_mean = torch.cat(tgt_mean, dim=0)
        r_gt_batch = torch.cat(r_gt, dim=0)
        info_batchids = torch.nn.utils.rnn.pad_sequence(info, batch_first=True, padding_value=pad_token_id)
        num_nodes = []
        for hd in hetero_data:
            num_nodes.append(hd['triple'].x.shape[0])

        return post_batchids, resp_batchids, Batch.from_data_list(hetero_data), t_gt_batch, t_gt_mean, r_gt_batch, info_batchids, num_nodes


class KOI(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        node_type = ['triple', 'relation']
        meta_data = (['triple', 'relation'], [('triple', 'contains', 'relation'), ('relation', 'belongsto', 'triple'), ('triple', 'next_klg', 'triple'),
                                              ('relation', 'next_rel', 'relation')])
        self.kg_encoder = HGT(self.args.hidden_size, self.args.hidden_size, self.args.graph_heads, self.args.graph_layers, node_type_list=node_type, metadata=meta_data)

        self.tri_brige = nn.Linear(self.args.hidden_size + self.args.tri_embed, self.args.tri_embed)
        self.rel_brige = nn.Linear(self.args.hidden_size + self.args.rel_embed, self.args.rel_embed)
        self.kg_gru = nn.GRU(input_size=self.args.tri_embed, hidden_size=self.args.gru_hidden, num_layers=self.args.gru_layers, bidirectional=True,
                             batch_first=True)
        self.rel_gru = nn.GRU(input_size=self.args.rel_embed, hidden_size=self.args.gru_hidden, num_layers=self.args.gru_layers, bidirectional=True,
                              batch_first=True)
        self.klg_attn = nn.MultiheadAttention(embed_dim=self.args.tri_embed, num_heads=self.args.graph_heads)
        self.tri_pred = nn.Linear(self.args.tri_embed, self.args.tri_vocab)
        self.bce_criterion = nn.BCEWithLogitsLoss()
        self.tri_gt_brige = nn.Linear(self.args.tri_embed, self.args.hidden_size)
        self.tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
        self.bart_dec = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese", add_cross_attention=False)
        self.bart_info = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")

        self.result_id = 0
        self.vocab_size = self.tokenizer.vocab_size
        self.all_vocab_size = self.vocab_size
        self.unk_id = self.tokenizer.unk_token_id

    def _prepare_input(self, data):
        input_ids = data
        attention_mask = (data != self.tokenizer.pad_token_id)
        return input_ids, attention_mask

    def forward(self, post_batchids, resp_batchids, hetero_data, t_gt_batch, t_gt_mean, r_gt_batch, info_batch, num_nodes, is_Train=True):
        post_ids, post_mask = self._prepare_input(post_batchids)
        resp_ids, resp_mask = self._prepare_input(resp_batchids)
        info_ids, info_mask = self._prepare_input(info_batch)
        KHGT_output = self.kg_encoder(hetero_data.x_dict, hetero_data.edge_index_dict)

        """所有结点向量残差+变回原维度600"""
        triple_residual = torch.cat((hetero_data['triple'].x, KHGT_output['triple']), 1)
        triple_embedding = self.tri_brige(triple_residual)
        relation_residual = torch.cat((hetero_data['relation'].x, KHGT_output['relation']), 1)
        relation_embedding = self.rel_brige(relation_residual)

        """把tri和rel变成batch形式"""
        tri_bc_lst, rel_bc_lst = [], []
        post_num = 0
        for i in range(self.args.batch_size):
            if num_nodes[i] == 0:
                triple_sigle = torch.zeros([1, self.args.tri_embed]).to("cuda:%d" % self.args.device_id)
                relation_single = torch.zeros([1, self.args.rel_embed]).to("cuda:%d" % self.args.device_id)
            else:
                triple_sigle = triple_embedding[post_num:post_num + num_nodes[i], :]
                relation_single = relation_embedding[post_num:post_num + num_nodes[i], :]
            post_num += num_nodes[i]
            tri_bc_lst.append(triple_sigle)
            rel_bc_lst.append(relation_single)
        tri_bc_embedding = torch.nn.utils.rnn.pad_sequence(tri_bc_lst, batch_first=True, padding_value=0).to("cuda:%d" % self.args.device_id)
        rel_bc_embedding = torch.nn.utils.rnn.pad_sequence(rel_bc_lst, batch_first=True, padding_value=0).to("cuda:%d" % self.args.device_id)
        # print(tri_bc_embedding)

        """用GRU预测下一步triple和relation"""
        tri_gru_out, tri_gru_state = self.kg_gru(tri_bc_embedding, torch.zeros(2 * self.args.gru_layers, self.args.batch_size, self.args.gru_hidden).to("cuda:%d" % self.args.device_id))
        tri_gru_out = tri_gru_out[:, -1:, :]
        rel_gru_out, rel_gru_state = self.rel_gru(rel_bc_embedding, torch.zeros(2 * self.args.gru_layers, self.args.batch_size, self.args.gru_hidden).to("cuda:%d" % self.args.device_id))
        rel_n = rel_gru_out[:, -1:, :]
        """预测的rel和所有的tri做attn"""
        klg_attn_out, attn_output_weights = self.klg_attn(rel_n, tri_gru_out, tri_gru_out)

        """添加一个多标签多分类层"""
        tri_pred = self.tri_pred(klg_attn_out)
        tri_pred = tri_pred.squeeze(1)

        """计算triple和relation的loss"""
        kg_loss = self.bce_criterion(tri_pred, t_gt_batch)

        enc_output = self.bart_dec(
            post_ids,
            attention_mask=post_mask
        )
        info_enc = self.bart_info(
            info_batch,
            attention_mask=info_mask
        )
        """将encoder_hidden与知识向量拼接+info，再decode"""
        info_embed = info_enc.encoder_last_hidden_state[:, -1:, :]
        tri_enc_embed = self.tri_gt_brige(klg_attn_out)
        enc_output_kg = torch.cat((enc_output.encoder_last_hidden_state, info_embed, tri_enc_embed), dim=1)

        encoder_output = ModelOutput()
        encoder_output['last_hidden_state'] = enc_output_kg
        encoder_output.last_hidden_state = enc_output_kg

        decoder_input_ids = resp_ids[:, :-1]
        decoder_attention_mask = (decoder_input_ids != self.tokenizer.pad_token_id)
        labels = resp_ids[:, 1:].clone()
        outputs = self.bart_dec(
            # input_ids=post_ids,
            # attention_mask=post_mask,
            encoder_outputs=encoder_output,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )
        logits = F.log_softmax(outputs.logits, dim=-1)
        gen_loss = outputs.loss
        total_loss = gen_loss + kg_loss

        if is_Train:
            return total_loss
        else:
            input_ids = torch.LongTensor([[101] for _ in range(self.args.batch_size)]).to("cuda:%d" % self.args.device_id)

            pred = self.bart_dec.generate(
                # inputs=post_ids,
                encoder_outputs=encoder_output,
                decoder_input_ids=input_ids,
                num_beams=5,
                max_length=150,
                # min_length=5,
                # return_dict_in_generate=True,
                # output_scores=True
            )       # [16,150]
            # sent = self.tokenizer.batch_decode(pred, skip_special_tokens=True)
            tri = (tri_pred, t_gt_batch)

            return logits, pred, tri

    def training_step(self, batch, batch_nb):
        loss = self.forward(*batch, is_Train=True)
        return loss

    def validation_step(self, batch, batch_nb):
        teacher_logits, gene_ids, tri = self.forward(*batch, is_Train=False)

        post_ids, resp_ids = batch[0], batch[1]
        post = self.tokenizer.batch_decode(post_ids, skip_special_tokens=True)
        resp = self.tokenizer.batch_decode(resp_ids, skip_special_tokens=True)
        gene = self.tokenizer.batch_decode(gene_ids, skip_special_tokens=True)

        metric1_data = {'resp_allvocabs': np.array(resp_ids.cpu()),
                        'resp_length': np.array([len(r) for r in resp_ids.cpu()]),
                        'gen_log_prob': np.array(teacher_logits.cpu())}
        metric2_data = {'gen': np.array(gene_ids.cpu()),
                        'resp_allvocabs': np.array(resp_ids.cpu())}

        return {'post': post, 'resp': resp, 'output': gene, 'tri': tri, 'm1': metric1_data, 'm2': metric2_data}

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        self.my_evaluate(outputs, self.args.gene_dir + '/test/test.txt')

    def validation_epoch_end(self, outputs):
        self.my_evaluate(outputs, self.args.gene_dir + '/val/val%d.txt' % self.result_id)
        self.result_id += 1

    def my_evaluate(self, outputs, filename):
        def get_teacher_forcing_metric(gen_log_prob_key="gen_log_prob", invalid_vocab=False):

            metric = MetricChain()
            metric.add_metric(PerplexityMetric(self, reference_allvocabs_key="resp_allvocabs", reference_len_key="resp_length",
                                               gen_log_prob_key=gen_log_prob_key, invalid_vocab=invalid_vocab))
            return metric

        def get_inference_metric(gen_key="gen"):
            metric = MetricChain()
            metric.add_metric(BleuCorpusMetric(self.tokenizer, gen_key=gen_key, reference_allvocabs_key="resp_allvocabs"))
            metric.add_metric(SingleTurnDistinct(self.tokenizer, gen_key=gen_key))
            return metric


        posts, resps = [], []
        endings = []
        metric1 = get_teacher_forcing_metric()
        metric2 = get_inference_metric()
        for item in outputs:
            metric1.forward(item['m1'])
            metric2.forward(item['m2'])

            posts.extend(item['post'])
            resps.extend(item['resp'])
            endings.extend(item['output'])

        res = metric1.close()
        res.update(metric2.close())
        res_print = list(res.items())
        print('----------------Evaluation-----------------')
        with open(filename, 'w', encoding='utf-8') as f:
            for m in res_print:
                f.write(str(m) + '\n')
                print(m)
            for i in range(len(endings)):
                f.write("post："+str(posts[i]).lower() + '\n')
                f.write("resp："+str(resps[i]).lower() + '\n')
                f.write("gene："+str(endings[i]).lower() + '\n')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        num_steps = self.args.dataset_size * self.args.epochs / self.args.grad_accum / self.args.batch_size
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup, num_training_steps=num_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _get_dataloader(self, key, is_train, batch_size):
        dataset = KdConvDataset(file_id=self.args.file_id, key=key, device_id=self.args.device_id)
        return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=0, drop_last=True, collate_fn=KdConvDataset.collate_fn)

    def train_dataloader(self):
        return self._get_dataloader(key='train', is_train=True, batch_size=self.args.batch_size)

    def val_dataloader(self):
        return self._get_dataloader(key='dev', is_train=False, batch_size=self.args.batch_size)

    def test_dataloader(self):
        return self._get_dataloader(key='test', is_train=False, batch_size=self.args.batch_size)


def main(args):
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dir,
        filename="weights_{epoch:03d}-{val_loss:.4f}.h5",
        save_top_k=3,
        verbose=True,
        monitor='loss',
        mode='min',
        # save_weights_only=True,
        save_last=True,
    )

    print(args)
    model = KOI(args)
    model.load_state_dict(torch.load('save_model/last.ckpt', map_location='cpu')['state_dict'])

    # args.dataset_size = 17389   # film:27550  # music:17999   # travel:17389

    trainer = pl.Trainer(
        gpus=[args.device_id],
        max_epochs=args.epochs,
        accumulate_grad_batches=args.grad_accum,
        track_grad_norm=-1,         # -1 no tracking. Otherwise tracks that p-norm.
        replace_sampler_ddp=False,
        # val_check_interval=0.25,       # 1 for debug, 0.25 for training
        check_val_every_n_epoch=1,
        # fast_dev_run=True,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback],
    )
    if not args.test:
        trainer.fit(model)
    else:
        trainer.test(model)


def add_model_specific_args(parser):
    parser.add_argument("--file_id", type=str, default='data')
    parser.add_argument("--save_dir", default="save_model", type=str, required=False)
    parser.add_argument("--gene_dir", default="output", type=str, required=False)

    parser.add_argument("--device_id", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=2, help="number of gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Maximum learning rate")
    parser.add_argument("--warmup", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--debug", action='store_true', help="debug run")

    parser.add_argument("--hidden_size", type=int, default=768, help="the hidden size of HGT")
    parser.add_argument("--graph_heads", type=int, default=8, help="Multi-head of Heterogeneous Graph Transformer")
    parser.add_argument("--graph_layers", type=int, default=1, help="GNN layers")
    parser.add_argument("--gru_layers", type=int, default=1, help="GRU layers")
    parser.add_argument("--gru_hidden", type=int, default=300, help="GRU hidden states")
    parser.add_argument("--tri_embed", type=int, default=600, help="the hidden size of triple")
    parser.add_argument("--rel_embed", type=int, default=200, help="the hidden size of relation")
    parser.add_argument("--tri_vocab", type=int, default=9815, help="the vocab size of triple")    # f79609 m46527 t9815
    parser.add_argument("--dataset_size", type=int, default=17389, help="the size of dataset")     # film:27550 music:17999 travel:17389

    parser.add_argument("--test", action='store_true', help="Test only, no training")

    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="model.py")
    parser = add_model_specific_args(parser)
    args = parser.parse_args()
    main(args)