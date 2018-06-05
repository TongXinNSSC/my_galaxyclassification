import pandas as pd
import pprint
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
def choose_clean_sample(dir):
    train_solution_df=pd.read_csv(dir)
    index_id = train_solution_df['GalaxyID']
    f_smooth = train_solution_df['Class1.1']
    f_completely_round = train_solution_df['Class7.1']
    f_in_between = train_solution_df['Class7.2']
    f_cigar_shaped = train_solution_df['Class7.3']
    f_features_disk = train_solution_df['Class1.2']
    f_edge_on_yes = train_solution_df['Class2.1']
    f_edge_on_no = train_solution_df['Class2.2']
    f_spiral_yes = train_solution_df['Class4.1']
    #print(index_id[0])
    # print(f_smooth.loc[0])
    # pprint.pprint(f_smooth)
    label = {}
    # label_id = {}
    for index in train_solution_df.index:
        if f_smooth.loc[index] >= 0.469 and f_completely_round.loc[index] >= 0.5:
         #   label.append([index, 0])
            label[index] = 0
        elif f_smooth.loc[index] >= 0.469 and f_in_between.loc[index] >= 0.5:
           # label.append([index, 1])
            label[index] = 1
        elif f_smooth.loc[index] >= 0.469 and f_cigar_shaped.loc[index] >= 0.5:
           # label.append([index, 2])
            label[index] = 2
        elif f_features_disk.loc[index] >= 0.430 and f_edge_on_yes.loc[index] >=0.602:
           # label.append([index, 3])
           label[index] = 3
        elif f_features_disk.loc[index] >= 0.430 and f_edge_on_no.loc[index] >= 0.715 and f_spiral_yes.loc[index] >= 0.619:
           # label.append([index, 4])
           label[index] = 4
        else:
            pass
  #  print(len(label))
    picture = []
    picture_label = []
    for key,value in label.items():
        key_id = index_id[key]
        # label_id[key_id] = value
        # print(len(label_id))
        picture.append('data/galaxy/images_training_rev1/'+str( key_id)+'.jpg ')
        picture_label.append(value)
    # draw = mpimg.imread('./'+ picture[0]) #读取图片
    # plt.imshow(draw)  # 显示图片
    # plt.axis('off')  # 不显示坐标轴
    # plt.show()
    data = {'root':picture,'label':picture_label,}
    f_data = pd.DataFrame(data)
    cols = f_data.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    f_data = f_data[cols]
    #print(f_data.head(3))
    #圆形星系
    data_round = f_data[f_data['label'] == 0]
    data_round = data_round.reset_index(drop=True)
    # print(data_round.shape)
    ind_list_r = list(range(data_round.shape[0]))
    ind_sample_r = random.sample(ind_list_r,round(data_round.shape[0]*0.9))
    # print("ind_sanmple_r : ",len(ind_sample_r)) #7592
    ind_rest_r = [x for x in ind_list_r if x not in ind_sample_r]
    # print("ind_rest_r : ", len(ind_rest_r))   # 844
    train_round = pd.DataFrame()
    test_round = pd.DataFrame()
    # print("data_round.index ",len(data_round.index))  #8436
    # print("data_round.index ",data_round.index)
    for index in ind_sample_r:
        train_round = train_round.append(data_round.loc[[index]], ignore_index=True)
    # print(" train_round ", train_round .shape)
    for index in ind_rest_r:
        test_round = test_round.append(data_round.loc[[index]], ignore_index=True)
    # drawname = test_round.to_string(header=False,index=False).split()[0]
    # print('drawname :'+drawname)
    # draw = mpimg.imread(drawname) #读取图片
    # plt.imshow(draw)  # 显示图片
    # plt.axis('off')  # 不显示坐标轴
    # plt.show()

    # print("test round ", test_round.shape)
    #middle
    data_middle = f_data[f_data['label'] == 1]
    data_middle = data_middle.reset_index(drop=True)
   # print(data_middle.shape)
    ind_list_m = list(range(data_middle.shape[0]))
    ind_sample_m = random.sample(ind_list_m,round(data_middle.shape[0]*0.9))
    ind_rest_m = [x for x in ind_list_m if x not in ind_sample_m]
    train_middle = pd.DataFrame()
    test_middle = pd.DataFrame()
    for index in ind_sample_m:
         train_middle = train_middle.append(data_middle.loc[[index]], ignore_index=True)
    for index in ind_rest_m:
         test_middle = test_middle.append(data_middle.loc[[index]], ignore_index=True)
   # print("middle ", test_middle.shape)
    #cigar
    data_cigar = f_data[f_data['label'] == 2]
    data_cigar = data_cigar.reset_index(drop=True)
    ind_list_c = list(range(data_cigar.shape[0]))
    ind_sample_c = random.sample(ind_list_c, round(data_cigar.shape[0] * 0.9))
    ind_rest_c = [x for x in ind_list_c if x not in ind_sample_c]
    train_cigar = pd.DataFrame()
    test_cigar = pd.DataFrame()
    for index in ind_sample_c:
        train_cigar = train_cigar.append(data_cigar.loc[[index]], ignore_index=True)
    for index in ind_rest_c:
        test_cigar = test_cigar.append(data_cigar.loc[[index]], ignore_index=True)
  #  print("cigar ", test_cigar.shape)
  #lateral
    data_lateral = f_data[f_data['label'] == 3]
    data_lateral = data_lateral.reset_index(drop=True)
    ind_list_l = list(range(data_lateral.shape[0]))
    ind_sample_l = random.sample(ind_list_l, round(data_lateral.shape[0]*0.9))
    ind_rest_l = [x for x in ind_list_l if x not in ind_sample_l]
    train_lateral = pd.DataFrame()
    test_lateral = pd.DataFrame()
    for index in ind_sample_l:
        train_lateral = train_lateral.append(data_lateral.loc[[index]], ignore_index=True)
    for index in ind_rest_l:
        test_lateral = test_lateral.append(data_lateral.loc[[index]], ignore_index=True)
   # print("lateral ",test_lateral.shape)
    # spiral
    data_spiral  = f_data[f_data['label'] == 4]
    data_spiral = data_spiral.reset_index(drop=True)
    ind_list_s = list(range(data_spiral.shape[0]))
    ind_sample_s = random.sample(ind_list_s, round(data_spiral.shape[0] * 0.9))
    ind_rest_s = [x for x in ind_list_s if x not in ind_sample_s]
    train_spiral = pd.DataFrame()
    test_spiral = pd.DataFrame()
    for index in ind_sample_s:
        train_spiral = train_spiral.append(data_spiral.loc[[index]], ignore_index=True)
    for index in ind_rest_s:
        test_spiral = test_spiral.append(data_spiral.loc[[index]], ignore_index=True)
   # print("spiral",test_spiral.shape)
   # print(data_spiral.head(3))
    train_clean_data = train_round.append([train_middle,train_cigar,train_lateral,train_spiral])
    train_clean_data = train_clean_data.sample(frac=1).reset_index(drop=True)
    test_clean_data = test_round.append([test_middle, test_cigar, test_lateral, test_spiral])
    test_clean_data = test_clean_data.sample(frac=1).reset_index(drop=True)
   # print('train ',train_clean_data.shape)
    print('test ', test_clean_data.shape)

#返回的f_data 是 dataframe，第一列叫‘root'，第二列叫"label"
    return train_clean_data, test_clean_data


if __name__ == '__main__':
    dir = 'data/galaxy/training_solutions_rev1/training_solutions_rev1.csv'
    train_clean_sample, test_clean_sample = choose_clean_sample(dir)
    filename1 = 'train_list.txt'
    filename2 = 'test_list.txt'
    train_clean_sample_str = train_clean_sample.to_string(header=False,index=False)
    test_clean_sample_str = test_clean_sample.to_string(header=False,index=False)
    with open(filename1,'w') as file:
        file.write(train_clean_sample_str)
    with open(filename2, 'w') as file:
        file.write(test_clean_sample_str)