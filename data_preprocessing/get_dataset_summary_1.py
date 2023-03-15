from os import listdir, mkdir 
from os.path import join,exists
import numpy as np
from PIL import Image
from tqdm import tqdm 
import pickle
import pandas as pd 
import shutil
from joblib import Parallel, delayed
import sys 
from pathlib import Path
sys.path.append(str(Path('..').absolute().parent))
from path_variables import *




def rename_img(img_name):
    new_img_name = img_name.replace(' ','')
    new_img_name = new_img_name.replace('.jpg','')
    new_img_name = new_img_name.replace('.JPG','')
    new_img_name += '.jpg'
    return new_img_name 

def generate_summary_per_folder(f):
    summary = []
    discarded_images = []
    for social_net in SOCIAL_NETWORKS:
        # {$BOLOGNA_PATH}/{$f}/{$social_net}
        social_network_path = join(BOLOGNA_PATH, f, social_net)
        if  exists(social_network_path):
            # {$BOLOGNA_PATH}/{$f}/{$social_net}/downloaded

            if exists(join(social_network_path, "downloaded")):
                downloaded_image_path = join(social_network_path, "downloaded")
                downloaded = True
            else:
                downloaded_image_path = social_network_path
                downloaded = False

            if exists(downloaded_image_path):
                # existing cameras: ofront, drear
                cameras = listdir(downloaded_image_path)
                for cam in cameras:
                    #  {$BOLOGNA_PATH}/{$f}/{$social_net}/downloaded/{$cam}  i.e dfront or orear
                    images_path = join(downloaded_image_path, cam)
                    for img_name in tqdm(listdir(images_path)):
                        # {$BOLOGNA_PATH}/{$f}/{$social_net}/downloaded/{$cam}/x.jpg
                        try:
                            current_img_path = join(images_path, img_name)
                            current_img = np.asarray(Image.open(current_img_path).convert('RGB'))/255
                            width, height, channels = current_img.shape
                            
                            
                            current_mean = np.mean(current_img, axis=(0, 1))
                            current_std = np.std(current_img, axis=(0,1))

                            new_image_name = f+"_"+social_net+"_"+cam+"_"+img_name

                            new_image_name = rename_img(new_image_name)

                            shutil.copy(current_img_path, join(out_images_path, new_image_name))
                            summary.append([f, social_net, downloaded, SOCIAL_NETWORKS_DICT[social_net], cam, 
                                            current_img_path, new_image_name, width, height, channels])
                            
                        except:
                            print("Discarded Image")
                            discarded_images.append(join(images_path, img_name))

    return summary, discarded_images 



if __name__ == "__main__":
 
    if not exists(FACIL_BOLOGNA_PATH):
        mkdir(FACIL_BOLOGNA_PATH)
 

    info_path = join(FACIL_BOLOGNA_PATH, "Info")
    if not exists(info_path):
        mkdir(info_path)
    else:
        print("Info Already Exists")
        # sys.exit("Info Already Exists")

    out_images_path = join(FACIL_BOLOGNA_PATH, "Images")

    if not exists(out_images_path):
        mkdir(out_images_path)
    else:
        print("Images Already Exists")
        #sys.exit("Images Already Exists")


    SOCIAL_NETWORKS_DICT = {"Facebook":0, 
                            "Flickr":1, 
                            "Google+":2, 
                            "GPhoto":3,
                            "Instagram":4,
                            "LinkedIn":5,
                            "Pinterest":6, 
                            "QQ":7, 
                            "Telegram":8,
                            "Tumblr":9,
                            "Twitter":10,
                            "Viber":11,
                            "VK":12,
                            "WeChat":13,
                            "WhatsApp":14,
                            "Wordpress":15}
        
    SOCIAL_NETWORKS = list(SOCIAL_NETWORKS_DICT.keys())
    
    folders =  listdir(BOLOGNA_PATH)

    result =  Parallel(n_jobs=len(folders))(delayed(generate_summary_per_folder)(f) for f in tqdm(folders))
    final_summary, final_discarded_images = [],[]
    # result is a list of tuple:  [(summary_f1, discarded_images_f1), ..., (summary_f19, discarded_images_f19)]
    for r in result: 
        final_summary.extend(r[0]) 
        final_discarded_images.extend(r[1])
                        
    file = open(join(info_path,'discarded_images.pkl'), 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(final_discarded_images, file)
    file.close()

    summary_df = pd.DataFrame(final_summary, columns = ['Folder', 'SocialNetwork', 'Downloaded','SocialNetworkID', 'Camera', 'OriginalImagePath',
                                                    'NewImageName',"Width", "Height", "Channels"])

    summary_df.to_csv(join(info_path,"summary.csv"), index=False)


 





                          



