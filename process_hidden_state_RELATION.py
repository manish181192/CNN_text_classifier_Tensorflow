f = open('test_results/hidden_states_results_10_02')

lines = f.readlines()
relation = {}
for i, line in enumerate(lines):

    split = line.split('\t')
    r = split[1]
    # pattern = split[0]
    emb = split[0]+"\t"+split[2]

    if relation.has_key(r):
        relation[r].append(emb)
    else:
        list = []
        list.append(emb)
        relation[r] = list

for key in relation:
    print key
    fw = open(str(key).replace('/','@') + '_embeddings_Test', 'w')
    for emb in relation[key]:
        fw.write(key+"\t"+emb+"\n")
    fw.close()



