import os
import numpy as np
import pickle

stat = [[[],[],[]],[[],[],[]],[[],[],[]]]

for i in range(1,10):
    i=1
    dir = './log/03-02' 
    for dataset_idx, dataset in enumerate(['wiki_ied_bombings','wiki_mass_car_bombings','suicide_ied']): # 
        test_etm_best_etm_F1_list = []
        test_etm_best_esm2_F1_list = []
        test_etm_best_esm3_F1_list = []

        test_esm3_best_etm_F1_list = []
        test_esm3_best_esm2_F1_list = []
        test_esm3_best_esm3_F1_list = []

        test_best_dev_index_etm_F1_list = []
        test_best_dev_index_esm2_F1_list = []
        test_best_dev_index_esm3_F1_list = []

        test_avg_etm_F1_list = []
        test_avg_esm2_F1_list = []
        test_avg_esm3_F1_list = []

        for seed in [2023,2024,2025,2026,2027]:#,2028,2029,2030,2031,2032]:
            if i==1:
                file_path = os.path.join(dir,dataset+'_seed='+str(seed)+'_main.log')
            else:
                file_path = os.path.join(dir,dataset+'_seed='+str(seed)+'_random_{}.log'.format(i))
            with open(file_path,"r") as f:
                lines = f.readlines()
                test_etm_best_etm_F1 = float(lines[-20].split('=')[-1].strip())
                test_etm_best_esm2_F1 = float(lines[-19].split('=')[-1].strip())
                test_etm_best_esm3_F1 = float(lines[-18].split('=')[-1].strip())

                test_esm3_best_etm_F1 = float(lines[-13].split('=')[-1].strip())
                test_esm3_best_esm2_F1 = float(lines[-12].split('=')[-1].strip())
                test_esm3_best_esm3_F1 = float(lines[-11].split('=')[-1].strip())

                test_best_dev_index_etm_F1 = float(lines[-4].split('=')[-1].strip())
                test_best_dev_index_esm2_F1 = float(lines[-3].split('=')[-1].strip())
                test_best_dev_index_esm3_F1 = float(lines[-2].split('=')[-1].strip())

                # test_avg_etm_F1
                for line in lines:
                    if 'test_avg_etm_F1' in line:
                        test_avg_etm_F1 = float(line.split('=')[1].strip())
                    if 'test_avg_esm2_F1' in line:
                        test_avg_esm2_F1 = float(line.split('=')[1].strip())
                    if 'test_avg_esm3_F1' in line:
                        test_avg_esm3_F1 = float(line.split('=')[1].strip())

                test_avg_etm_F1_list.append(test_avg_etm_F1)
                test_avg_esm2_F1_list.append(test_avg_esm2_F1)
                test_avg_esm3_F1_list.append(test_avg_esm3_F1)

                test_etm_best_etm_F1_list.append(test_etm_best_etm_F1)
                test_etm_best_esm2_F1_list.append(test_etm_best_esm3_F1)
                test_etm_best_esm3_F1_list.append(test_etm_best_esm3_F1)

                test_esm3_best_etm_F1_list.append(test_esm3_best_etm_F1)
                test_esm3_best_esm2_F1_list.append(test_esm3_best_esm2_F1)
                test_esm3_best_esm3_F1_list.append(test_esm3_best_esm3_F1)

                test_best_dev_index_etm_F1_list.append(test_best_dev_index_etm_F1)
                test_best_dev_index_esm2_F1_list.append(test_best_dev_index_esm2_F1)
                test_best_dev_index_esm3_F1_list.append(test_best_dev_index_esm3_F1)
        print(dataset)
        print('test_etm_best_etm_F1= '+'{:.3f}'.format(np.mean(test_etm_best_etm_F1_list))+' {:.3f}'.format(np.std(test_etm_best_etm_F1_list)))
        print('test_etm_best_esm2_F1= '+'{:.3f}'.format(np.mean(test_etm_best_esm2_F1_list))+' {:.3f}'.format(np.std(test_etm_best_esm2_F1_list)))
        print('test_etm_best_esm3_F1= '+'{:.3f}'.format(np.mean(test_etm_best_esm3_F1_list))+' {:.3f}'.format(np.std(test_etm_best_esm3_F1_list)))
        print()
        print('test_esm3_best_etm_F1= '+'{:.3f}'.format(np.mean(test_esm3_best_etm_F1_list))+' {:.3f}'.format(np.std(test_esm3_best_etm_F1_list)))
        print('test_esm3_best_esm2_F1= '+'{:.3f}'.format(np.mean(test_esm3_best_esm2_F1_list))+' {:.3f}'.format(np.std(test_esm3_best_esm2_F1_list)))
        print('test_esm3_best_esm3_F1= '+'{:.3f}'.format(np.mean(test_esm3_best_esm3_F1_list))+' {:.3f}'.format(np.std(test_esm3_best_esm3_F1_list)))
        print()
        print('test_best_dev_index_etm_F1= '+'{:.3f}'.format(np.mean(test_best_dev_index_etm_F1_list))+' {:.3f}'.format(np.std(test_best_dev_index_etm_F1_list)))
        print('test_best_dev_index_esm2_F1= '+'{:.3f}'.format(np.mean(test_best_dev_index_esm2_F1_list))+' {:.3f}'.format(np.std(test_best_dev_index_esm2_F1_list)))
        print('test_best_dev_index_esm3_F1= '+'{:.3f}'.format(np.mean(test_best_dev_index_esm3_F1_list))+' {:.3f}'.format(np.std(test_best_dev_index_esm3_F1_list)))
        print()
        stat[dataset_idx][0].append(np.mean(test_avg_etm_F1_list))
        stat[dataset_idx][1].append(np.mean(test_avg_esm2_F1_list))
        stat[dataset_idx][2].append(np.mean(test_avg_esm3_F1_list))
with open('./stat.pkl','wb') as file:
    pickle.dump(stat,file)

# pickle.load(open("./kairos_ontology.pkl", "rb"))
print()
