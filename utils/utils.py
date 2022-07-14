import jieba
import orjson as js
import json
import torch
import numpy as np


def only_triple():
    """加pad符号后的id"""
    klg2id = {}
    fw = open("../data/travel/triple2id.txt", 'w', encoding='utf-8')
    with open('../data/travel/klg2id.txt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            k = tuple(line.strip().split('\t'))
            if k not in klg2id.keys():
                klg2id[k] = i
                triple = str(int(k[0])+1) + '\t' + str(int(k[1])+1) + '\t' + str(int(k[2])+1) + '\n'
                fw.write(triple)
    print("已生成triple2id")


def get_triple_embed():
    """加pad符后的embedding"""
    with open('../data/travel/embed_transR.vec', 'rb') as f:
        data = js.loads(f.read())
        entity_embed = [[0.0]*200] + data["ent_embeddings.weight"]
        relation_embed = [[0.0]*200] + data["rel_embeddings.weight"]
    triple_embed = [[0.0]*600]
    with open('../data/travel/triple2id.txt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            h, r, t = line.strip().split('\t')
            he = entity_embed[int(h)]
            re = relation_embed[int(r)]
            te = entity_embed[int(t)]
            hrte = he + re + te
            # print(len(hrte))
            triple_embed.append(hrte)
        # triple_embed = np.array(triple_embed, dtype=np.float32)
    # print(len(triple_embed))
    with open('../data/travel/triple_embed.json', 'wb') as fw:
        fw.write(js.dumps(triple_embed))
        print("triple_embed写入成功")


def next_triple():
    entity2id = {}
    with open('../data/entity2id.txt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            e, idx = line.strip().split('\t')
            entity2id[e] = int(idx)

    relation2id = {}
    with open('../data/relation2id.txt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            r, idx = line.strip().split('\t')
            relation2id[r] = int(idx)

    tri2id = {}
    with open('../data/triple2id.txt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            k = tuple([int(x) for x in line.strip().split('\t')])
            if k not in tri2id.keys():
                tri2id[k] = i
            # else:
                # print("重复", k, tri2id[k], i)
    print("triple2id=", len(tri2id))

    datas = json.load(open('../data/train.json', encoding='utf8'))
    k_train, r_train = [], []
    for data in datas:
        messages = data['messages']
        k_dial, r_dial = [], []
        for message in messages:
            sent = message['message']
            k_sent, r_sent = [], []
            if 'attrs' in message:
                for attr in message['attrs']:
                    if attr['attrname'] == 'Information':
                        continue
                    h = entity2id[attr['name'].replace('【', '').replace('】', '')]
                    r = relation2id[attr['attrname'].replace('【', '').replace('】', '')]
                    t = entity2id[attr['attrvalue'].replace('【', '').replace('】', '')]

                    if (h, r, t) in tri2id.keys():
                        k_sent.append(tri2id[(h, r, t)])
                        r_sent.append(r)
                    else:
                        print("没有",(h, r, t))

                k_dial.append(k_sent)
                r_dial.append(r_sent)
        k_train.append(k_dial)
        r_train.append(r_dial)

    # 知识-知识边 前节点tri_past，后节点tri_next
    tri_past, tri_next = [], []
    for k_dial in k_train:
        for i, k_sent in enumerate(k_dial):
            if i == len(k_dial) - 1:
                break
            for j1, t1 in enumerate(k_sent):
                for j2, t2 in enumerate(k_dial[i+1]):
                    tri_past.append(t1)
                    tri_next.append(t2)
    with open('data/triple_next.json', 'wb') as ft:
        ft.write(js.dumps([tri_past, tri_next]))
        print("triple_next写入成功")

    # 关系-关系边
    re_past, re_next = [], []
    for r_dial in r_train:
        for i, r_sent in enumerate(r_dial):
            if i == len(r_dial) - 1:
                break
            for j1, r1 in enumerate(r_sent):
                for j2, r2 in enumerate(r_dial[i+1]):
                    re_past.append(r1)
                    re_next.append(r2)
    with open('data/relation_next.json', 'wb') as ft:
        ft.write(js.dumps([re_past, re_next]))
        print("relation_next写入成功")


def process_load(file_id):
    print("Loading triple vectors...")
    with open('%s/triple_embed.json' % file_id, 'rb') as ft:
        triple_embed = js.loads(ft.read())
    print('triple=', len(triple_embed))

    print("Loading entity and relation vectors...")
    with open('%s/embed_transR.vec' % file_id, 'rb') as f:
        data = js.loads(f.read())
        # entity_embed = [[0.0] * 200] + data["ent_embeddings.weight"]
        relation_embed = [[0.0] * 200] + data["rel_embeddings.weight"]

    # entity_embed = np.array(entity_embed, dtype=np.float32)
    # relation_embed = np.array(relation_embed, dtype=np.float32)

    print('relation=', len(relation_embed))
    return triple_embed, relation_embed


def get_klg2id(file_id):
    entity2id = {'_PAD_E': 0}
    with open('%s/entity2id.txt' % file_id, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            e, idx = line.strip().split('\t')
            entity2id[e] = int(idx) + 1
    print("entity2id=", len(entity2id))
    relation2id = {'_PAD_R': 0}
    with open('%s/relation2id.txt' % file_id, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            r, idx = line.strip().split('\t')
            relation2id[r] = int(idx) + 1
    print("relation2id=", len(relation2id))
    tri2id = {'_PAD_T': 0}
    with open('%s/triple2id.txt' % file_id, encoding='utf-8') as f:
        for i, line in enumerate(f):
            k = tuple([int(x) for x in line.strip().split('\t')])
            if k not in tri2id.keys():
                tri2id[k] = i + 1
            # else:
                # print("重复", k, tri2id[k], i)
    print("triple2id=", len(tri2id))
    kg2dialkg = {}
    with open('%s/kg2dialkg.txt' % file_id) as f:
        for i, line in enumerate(f):
            kid, did = line.strip().split('\t')
            kg2dialkg[int(kid)] = int(did)
    print("kg2dialkg=", len(kg2dialkg))
    return entity2id, relation2id, tri2id, kg2dialkg


def get_dial_kg():
    """
    从对话中提取涉及到的知识三元组（KG中有的实体对话中未提到，训练可能造成稀疏）
    """
    kg2dialkg, dialkg2id = {}, {}
    entity2id, relation2id, triple2id = get_klg2id()
    files = ['train', 'dev', 'test']
    i = 1
    for f in files:
        with open('data/travel/' + f + ".json", 'r', encoding='utf-8') as fj:
            all_lst = json.load(fj)
            fj.close()
        for dial in all_lst:
            messages = dial["messages"]
            for mess in messages:
                if "attrs" in mess.keys():
                    for attr in mess["attrs"]:
                        if attr['attrname'] != 'Information':
                            h = entity2id[attr['name'].replace('【', '').replace('】', '')]
                            r = relation2id[attr['attrname'].replace('【', '').replace('】', '')]
                            t = entity2id[attr['attrvalue'].replace('【', '').replace('】', '')]
                            if (h, r, t) not in dialkg2id:
                                dialkg2id[(h, r, t)] = i
                                i += 1
                        # attrname = attr["attrname"]  # 属性名
                        # attrvalue = attr["attrvalue"]  # 属性值
                        # headname = attr["name"]  # 头实体
                        # if attrname != 'Information':
                        #     if headname not in truple:
                        #         truple[headname] = [[headname, attrname, attrvalue]]
                        #     else:
                        #         if [headname, attrname, attrvalue] not in truple[headname]:     # 去重
                        #             truple[headname].append([headname, attrname, attrvalue])

    with open('data/travel/dialkg2id_noinfo.txt', 'w', encoding='utf8') as kg_dial:  # 记录对话中涉及的知识
        for key in dialkg2id:
            kg_dial.write(str(key[0])+'\t'+str(key[1])+'\t'+str(key[2])+'\n')
    with open('data/travel/kg2dialkg.txt', 'w') as fw:
        for key in dialkg2id:
            kgid = triple2id[key]
            kg2dialkg[kgid] = dialkg2id[key]
            fw.write(str(kgid)+'\t'+str(dialkg2id[key])+'\n')

    return dialkg2id, kg2dialkg


def get_wordvector():
    def count_token(tokens):
        for token in tokens:
            vocab[token] = vocab[token] + 1 if token in vocab else 1

    vocab = {}
    for key in ['train', 'dev', 'test']:
        datas = json.load(open('%s/%s.json' % ('data/travel', key), encoding='utf8'))
        for data in datas:
            messages = data['messages']
            for message in messages:
                if 'attrs' in message:
                    for attr in message['attrs']:
                        if attr['attrname'] == 'Information':
                            info_sent = attr['attrvalue']
                            count_token(info_sent)
                            continue
                        h = jieba.lcut(attr['name'].replace('【', '').replace('】', ''))
                        r = jieba.lcut(attr['attrname'].replace('【', '').replace('】', ''))
                        t = jieba.lcut(attr['attrvalue'].replace('【', '').replace('】', ''))
                        k = h + r + t
                        count_token(h + r + t)

    word2id = {w: i + 1 for i, w in enumerate(vocab.keys())}
    word2id['<pad>'] = 0
    json.dump(word2id, open('%s/word2id.json' % 'data/travel', 'w', encoding='utf8'))
    print("已写入词嵌入：", len(word2id))      # f:11452, m:6228, t:6163

if __name__ == '__main__':
    """注意：训练前需要修改文件夹路径，film/music/travel"""
    # only_triple()
    # get_triple_embed()

    # entity2id, relation2id, tri2id = get_klg2id()
    # triple_embed, relation_embed = process_load()

    get_wordvector()

    # dialkg2id, kg2dialkg = get_dial_kg()
