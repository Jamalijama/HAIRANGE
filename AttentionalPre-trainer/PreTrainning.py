from transformer import *

fixed_value = 0.05

# sequences before 2020 for training single model
#file_name = 'before2020_29634_'

#file_name = 'IAV_reassortant_indp_'

# sequences after 2020 for testing single model
# file_name = 'after2020_27016_'

# sequences for simulated sequence reassortment
# file_name = '23366_all_AvianSwine_'

# given reassort sequences
# file_name = 'AIV_reassort_84_'

#file_name = 'test_'

#file_name = 'AIV_all_8_avian_humanH3N2_'
#file_name = 'H7N9_H5N1_all_4_'
#file_name = 'AIV_all_8_avian_humanH1N1_'

file_name = 'AIV_all_8_a_hH1N1_aH3N2before2008_'

cds = ['pb2', 'pb1', 'pa', 'np']


def PreTrain(fixed_value, freq_path_in, freq_path_out):
    interval = 50
    trm_batch_size = 50

    all_inputs = np.load(freq_path_in, allow_pickle=True)
    seq_num = len(all_inputs)
    all_inputs = torch.FloatTensor(all_inputs)

    all_outputs = []
    model = Transformer(fixed_value)
    model = model.to(device)

    for i in range(0, seq_num, interval):
        all_input = all_inputs[i:i + interval].clone()
        loader = Data.DataLoader(MyDataSet_freq(all_input), trm_batch_size, False)
        for input in loader:
            input = input.to(device)
            output = model(input).to(device).detach().cpu()
            # output_ = output.clone()
            all_outputs.extend(output.numpy())

    array_input = np.array(all_outputs)
    np.save(freq_path_out, array_input, allow_pickle=True)


if __name__ == '__main__':
    for cdss in cds:
        freq_path_in = '../Res/npy/array_codon_freq_' + file_name + cdss + '.npy'
        freq_path_out = '../Res/npy_trm/array_codon_freq_' + file_name + cdss + '_trm.npy'
        print(freq_path_in)
        PreTrain(fixed_value, freq_path_in, freq_path_out)
