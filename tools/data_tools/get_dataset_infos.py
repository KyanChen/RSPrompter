import pickle


ann_data_prefix = 'I:/CodeRep/INRCls/data_list/UC'
data_prefix = 'I:/CodeRep/INRCls/results/EXP20220422_3'
ann_file = ann_data_prefix + '/train_list.txt'

# 48x
with open(ann_file) as f:
    samples = [x.strip().rsplit('.tif ', 1) for x in f.readlines()]

for filename in samples:
    with open(data_prefix+ '/' + results['img_info']['filename'], 'rb') as f:
        results['img'] = pickle.load(f)['modulations']
    results['img'] = np.array(results['img']).astype(np.float32).reshape(-1)