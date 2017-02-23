ep_rel = {}
lines = open('/home/rpothams/Documents/Gold_Patterns/Filtered/Train_Data/total_train_data_03_02','r').readlines()
rel_id = {}
id = 0
for l in lines:
    splt = l.split("\t")

    if l.startswith("REL$/") == True:
        relation = splt[0]
        entity1 = splt[1]
        entity2 = splt[2]
        enPair = entity1 + "\t" + entity2
        ep_rel[enPair] = relation
        if rel_id.has_key(relation) == False:
            rel_id[relation] = id
            id = id + 1

fw = open('resources/traindata_for_classifier_06_02','w')
for l in lines:
    splt = l.split("\t")
    if l.startswith("REL$/") == False:
        pattern = splt[0]
        entity1 = splt[1]
        entity2 = splt[2]
        enPair = entity1 + "\t" + entity2
        fw.write(pattern + "\t" + str(rel_id[ep_rel[enPair]]) + "\n")
fw.close()
lines = open('/home/rpothams/Documents/Gold_Patterns/Filtered/Cross_Validation_Data/total_CV_03_02','r').readlines()
fw = open('resources/total_CV_06_02_classifier','w')
for l in lines:
    splt = l.strip("\n").split("\t")
    fw.write(splt[0] + "\t"+str(rel_id[splt[1]]) + "\n")