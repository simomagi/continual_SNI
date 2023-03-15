from os.path import join 
import pandas as pd 
import sys 
from pathlib import Path
sys.path.append(str(Path('..').absolute().parent))
from path_variables import *


if __name__ == "__main__":
    info_path = join(FACIL_BOLOGNA_PATH,"Info")
    image_path = join(FACIL_BOLOGNA_PATH, "Images") 

    df = pd.read_csv(join(info_path,"summary.csv"))
    # Filter small Images than 256 x 256 
    df = df[(df.Width>= 256) & (df.Height>= 256)].reset_index(drop=True)
    # get only downloaded Images (Flickr  And Linkedin Are Removed)
    df = df[df.Downloaded==True].reset_index(drop=True)

    # re-map classes ranging from 0 to 13
    filtered_socialnet =   {"Facebook":0, 
                            "GPhoto":1,
                            "Google+":2, 
                            "Instagram":3,
                            "Pinterest":4, 
                            "QQ":5, 
                            "Telegram":6,
                            "Tumblr":7,
                            "Twitter":8,
                            "VK":9,
                            "Viber":10,
                            "WeChat":11,
                            "WhatsApp":12,
                            "Wordpress":13}        #"Flickr":1, "LinkedIn":5,

    df["SocialNetworkID"] = pd.Categorical(df.SocialNetwork, ordered=True).codes
    
    df['ImgId'] = df.index
    

    train_df = df[~df.Folder.isin([16,17,18,19])].reset_index(drop=True)
    test_df = df[df.Folder.isin([16,17,18,19])].reset_index(drop=True)

    valid_df =  train_df[train_df.Folder.isin([10,15])].reset_index(drop=True)
    train_df =  train_df[~train_df.Folder.isin([10,15])].reset_index(drop=True)


    train_img_name = [name for name in train_df.NewImageName.tolist()] 
    train_classes =  train_df.SocialNetworkID.tolist()
    train_ids = train_df.ImgId.tolist()

    train_ann = [[p, str(c), str(i)] for p, c, i in zip(train_img_name, train_classes, train_ids)]

 
    valid_img_name = [name for name in valid_df.NewImageName.tolist()] 
    valid_classes = valid_df.SocialNetworkID.tolist()
    val_ids = valid_df.ImgId.tolist()
    valid_ann = [[p, str(c), str(i)] for p, c, i  in zip(valid_img_name, valid_classes, val_ids)]
 

    test_img_name = [name for name in test_df.NewImageName.tolist()] 
    test_classes = test_df.SocialNetworkID.tolist()
    test_ids = test_df.ImgId.tolist()

    test_ann = [[p, str(c), str(i)] for p, c, i in zip(test_img_name, test_classes, test_ids)]
    test_ids = test_df.ImgId.tolist()

    with open(join(info_path, 'train.txt'),'w') as train_annotations_file:
        for item in train_ann:
            train_annotations_file.write(' '.join(item)+'\n')

    with open(join(info_path, 'val.txt'),'w') as valid_annotations_file:
        for item in valid_ann:
            valid_annotations_file.write(' '.join(item)+'\n')

    with open(join(info_path, 'test.txt'),'w') as test_annotations_file:
        for item in test_ann:
            test_annotations_file.write(' '.join(item)+'\n')






