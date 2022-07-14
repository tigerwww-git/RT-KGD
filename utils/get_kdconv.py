import json
import random


# entity2id.txt
def get_entity(entity):
    fent = open('kdconv/entity2id.txt', 'w', encoding='utf8')
    fent.write(str(len(entity)))
    e2id = {}
    for i, e in enumerate(entity):
        if e not in e2id:
            fent.write('\n')
            e2id[e] = len(e2id)
            fent.write(e + '\t' + str(e2id[e]))

    return e2id


# relation2id.txt
def get_relation(relation):
    frel = open('kdconv/relation2id.txt', 'w', encoding='utf8')
    frel.write(str(len(relation)))
    r2id = {}
    for i, r in enumerate(relation):
        if r not in r2id:
            frel.write('\n')
            r2id[r] = len(r2id)
            frel.write(r + '\t' + str(r2id[r]))

    return r2id


# train2id.txt
def get_train(entity, relation, data):
    e2id = get_entity(entity)
    r2id = get_relation(relation)
    data2id = []
    for d in data:
        if d[0] in e2id and d[1] in e2id and d[2] in r2id:
            data2id.append([e2id[d[0]], e2id[d[1]], r2id[d[2]]])

    randnum = random.sample(range(0, len(data)), int(len(data) / 56) * 6)  # 按照50:1:5的比例分为train、valid、test
    # train2id, test2id, valid2id = [], [], []
    ftrain = open('kdconv_travel/train2id.txt', 'w', encoding='utf8')
    ftrain.write(str(len(data) - len(randnum)) + '\n')
    ftest = open('kdconv_travel/test2id.txt', 'w', encoding='utf8')
    ftest.write(str(len(randnum) - int(len(randnum) / 6)) + '\n')
    fvalid = open('kdconv_travel/valid2id.txt', 'w', encoding='utf8')
    fvalid.write(str(int(len(randnum) / 6)) + '\n')
    for i, num in enumerate(randnum):
        if i % 6 == 0:
            fvalid.write(str(data2id[num][0]) + '\t' + str(data2id[num][1]) + '\t' + str(data2id[num][2]) + '\n')
        else:
            ftest.write(str(data2id[num][0]) + '\t' + str(data2id[num][1]) + '\t' + str(data2id[num][2]) + '\n')
    for j, dt in enumerate(data2id):
        if j not in randnum:
            ftrain.write(str(dt[0]) + '\t' + str(dt[1]) + '\t' + str(dt[2]) + '\n')


def clean_data(data):
    data = data.replace('【', '')
    data = data.replace('】', '')
    return data.strip()


def get_klg2id(entity, relation, data):
    fk = open('kdconv_travel/klg2id.txt', 'w', encoding='utf8')
    e2id = get_entity(entity)
    r2id = get_relation(relation)
    klg2id = {}
    for d in data:
        if d[0] in e2id and d[1] in e2id and d[2] in r2id:
            k = tuple([e2id[d[0]], r2id[d[2]], e2id[d[1]]])
            if k not in klg2id.keys():
                klg2id[k] = 1
                line = "\t".join(map(str, [e2id[d[0]], r2id[d[2]], e2id[d[1]]]))
                fk.write(line+'\n')
            # else:
                # print("头：", e2id[d[0]], d[0], "\t关系：", r2id[d[2]], d[2], "\t尾：", e2id[d[1]], d[1])

    print("triple2id：", len(klg2id))

def main():
    """
    将总的知识图谱进行编码，可以捕获并融入更多周边信息
    """
    with open('kb_travel.json', 'r', encoding='utf-8') as fkg:  # 总的KG，而不仅仅是对话中涉及的
        kb_dict = json.load(fkg)
        entity, relation, data = [], [], []

        for kb in kb_dict:
            h = clean_data(kb)
            if len(h) != 0 and h not in entity:
                entity.append(h)
            for kb_list in kb_dict[kb]:
                r = clean_data(kb_list[1])
                t = clean_data(kb_list[2])
                if len(r) != 0 and r not in relation:
                    if r == "Information":
                        continue
                    relation.append(r)
                if len(t) != 0 and t not in entity:
                    entity.append(t)
                data.append([h, t, r])

        print("entity length is ", len(entity))
        print("relation length is ", len(relation))
        print("triple length is ",len(data))
        get_train(entity, relation, data)
        get_klg2id(entity, relation, data)


if __name__ == '__main__':
    main()
