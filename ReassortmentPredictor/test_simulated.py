from module import *
from resnet_18_34 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data
#file_name = '23366_all_AvianSwine'
#file_name = 'test'
file_name = 'H7N9_H5N1_all_4'
freq_path = '../Res/npy_reassort/humanH3N2_AvianSwine_simulated_' + file_name + '_0531_2816.npy'
id_path = '../Res/res1/humanH3N2_AvianSwine_simulated_name_' + file_name + '_0531_2816.csv'

inputs = np.load(freq_path, allow_pickle=True)
seq_num = len(inputs)

df = pd.read_csv(id_path)
ids = df['name'].tolist()

inputs = torch.FloatTensor(inputs)
print(inputs.shape)

inputs = inputs.view(inputs.size()[0], -1, 128, 128)
mdl_batch_size = 1000
setup_seed(7)
loader = Data.DataLoader(MyDataSet_freqId(inputs, ids), mdl_batch_size, False, num_workers=0)
# load the model
model = ResNet34(2, 11)
model.load_state_dict(torch.load('../Res/pt/resnet_classification_500+0.03_10_resnet34_reassort_a_hH1N1_aH3N2_0.05_0701_2816.pt'))
model.to(device)

model.eval()
predictions = []
probs = []
idss = []
with torch.no_grad():
    for input, id in loader:
        input = input.to(device)
        preds = model(input).to(device)
        prob = F.softmax(preds, dim=1)
        top_p, top_class = prob.topk(1, dim=1)
        probs.extend(top_p)
        predictions.extend(top_class)
        idss.extend(list(id))

pred_lst = [pred_.item() for pred_ in predictions]
probs_lst = [prob.item() for prob in probs]

df = pd.DataFrame()
df['id'] = idss
df['pred'] = pred_lst
df['prob'] = probs_lst
# cds1
df.to_csv('../Res/prediction_result_csv/humanH3N2_simulated_' + file_name + '_a_hH1N1_aH3N2_10_0.05_0701_2816.csv', index=False)
