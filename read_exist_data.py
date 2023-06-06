import pickle
import os
def read_exist_data(args):
    saved_dict = pickle.load(open("./kairos_ontology.pkl", "rb"))[0]
    event_types_ontology = saved_dict['event_types']
    event_types_ontology_new = {"START": 0, "END": 1,"PAD": 69}
    for key, val in event_types_ontology.items():
        event_types_ontology_new[key] = val + 2
    event_type2id_dict = event_types_ontology_new
    dir = './'
    
    if True:
        for dataset in ['wiki_ied_bombings','wiki_mass_car_bombings','suicide_ied']:
            file_path = dir+'data/Wiki_IED_split/train/{}_train_pruned_new_no_iso_max_'.format(args.dataset) + str(args.max_n) + '_igraphs.pkl'
            train_data = pickle.load(open(file_path, 'rb'))
            train_data = [t for t,_ in train_data]
    else:
        file_path = dir+'data/Wiki_IED_split/train/{}_train_pruned_new_no_iso_max_'.format(args.dataset) + str(args.max_n) + '_igraphs.pkl'
        train_data = pickle.load(open(file_path, 'rb'))
        train_data = [t for t,_ in train_data]
    file_path = dir+'data/Wiki_IED_split/dev/{}_dev_pruned_new_no_iso_max_'.format(args.dataset) + str(args.max_n) + '_igraphs.pkl'
    dev_data = pickle.load(open(file_path, 'rb'))
    dev_data = [t for t,_ in dev_data]
    file_path = dir+'data/Wiki_IED_split/test/{}_test_pruned_new_no_iso_max_'.format(args.dataset) + str(args.max_n) + '_igraphs.pkl'
    test_data = pickle.load(open(file_path, 'rb'))
    test_data = [t for t,_ in test_data]
    return train_data,dev_data,test_data,event_type2id_dict