import torch
from glob import glob
import os
from PIL import Image
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from time import time
from argparse import ArgumentParser
import cv2
import json

import sys
sys.path.append('.')
from recog_model.structure.MPCount import DGModel_final
from rutils import denormalize, divide_img_into_patches, get_padding
from utils import read_json, get_subdirectories



def load_model(model_path, device):
    model = DGModel_final().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()

    return model

def covert_img(img_path, unit_size, device):
    assert os.path.exists(img_path), f'Image path {img_path} does not exist.'
    assert img_path.lower().endswith('.jpg') or img_path.lower().endswith('.jpeg') \
        or img_path.lower().endswith('.png'), 'Only support .jpg and .png image format.'
    
    img = Image.open(img_path).convert('RGB')
    
    if unit_size > 0:
        w, h = img.size
        new_w = (w // unit_size + 1) * unit_size if w % unit_size != 0 else w
        new_h = (h // unit_size + 1) * unit_size if h % unit_size != 0 else h

        padding, h, w = get_padding(h, w, new_h, new_w)

        img = F.pad(img, padding)
    img = F.to_tensor(img)
    img = F.normalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    img = img.unsqueeze(0).to(device)
    return img

@torch.no_grad()
def predict(model, img_path, device="cuda", unit_size=16, patch_size=3584, log_para=1000):
    img = covert_img(img_path, unit_size, device)
    
    h, w = img.shape[2:]
    ps = patch_size
    if h >= ps or w >= ps:
        pred_dmap = torch.zeros(1, 1, h, w)
        pred_count = 0
        img_patches, nh, nw = divide_img_into_patches(img, ps)
        for i in range(nh):
            for j in range(nw):
                patch = img_patches[i*nw+j]
                pred_dpatch = model(patch)[0]
                pred_dmap[:, :, i*ps:(i+1)*ps, j*ps:(j+1)*ps] = pred_dpatch
    else:
        pred_dmap = model(img)[0]
    pred_count = pred_dmap.sum().cpu().item() / log_para

    return img, pred_dmap.squeeze().cpu().numpy(), pred_count


if __name__=="__main__":
    model_name = "MPCount"
    config = read_json("recog_model/recong_model_config.jsonc")
    config_mp = config["crowd_model"][model_name]
    input_dir = config["input_dir"]
    saved_dir = config["saved_dir"]
    vis_dir = config["vis_dir"]

    
    # image_dir_paths = get_subdirectories(input_dir,2)
    model = load_model(config_mp["pretrained"], config_mp["device"])
    
    areas = sorted(os.listdir(input_dir))
    for area in areas:
        dates_dir_path = os.path.join(input_dir, area)
        dates = sorted(os.listdir(dates_dir_path))
        idx = 0
        for date in dates:
            area_timestamp_dict= dict()
            img_dir_path = os.path.join(input_dir, area, date)
            frame_names = sorted(os.listdir(img_dir_path))
            for frame_name in frame_names:
                frame_path = os.path.join(img_dir_path, frame_name)
                img, pred_dmap, pred_count =predict(model, frame_path, config_mp["device"], config_mp["unit_size"],config_mp["patch_size"], config_mp["log_para"])
                confidence = None
                time_str = frame_name.split("_")[-1].split(".")[0]
                hour = int(time_str[0:2])
                minute = int(time_str[2:])
                nminute = minute-minute%5
                ntimestamp = f'{date}{hour:02}{nminute:02}'
                area_timestamp_dict[ntimestamp] = {"id": idx,  "count": pred_count, "conf":confidence, "area": area,  "date": date, "time":time_str, "frame_name": frame_name, "model": model_name}
                idx+=1
                print(f'{frame_path}: {pred_count}')
                
                if vis_dir is not None:
                    os.makedirs(vis_dir, exist_ok=True)
                    denormed_img = denormalize(img)[0].cpu().permute(1, 2, 0).numpy()
                    fig = plt.figure(figsize=(10, 5))
                    ax_img = fig.add_subplot(121)
                    ax_img.imshow(denormed_img)
                    ax_img.set_title("Origin",y=-0.2)
                    ax_dmap = fig.add_subplot(122)
                    ax_dmap.imshow(pred_dmap)
                    ax_dmap.set_title("Heatmap",y=-0.2)
                    area, date, _ = frame_name.split("_")
                    # 为整个大图设置标题
                    fig.suptitle(f"Device: {area}   Date: {date}   Time: {time_str}", fontsize=16, y=0.75)

                    # 调整子图布局，避免标题重叠
                    plt.tight_layout(rect=[0, 0, 1, 0.9])

                    # import matplotlib
                    # matplotlib.image.imsave(os.path.join(vis_dir,f"{frame_name[:-4]}({int(pred_count)}).jpg"), pred_dmap)
                    plt.savefig(os.path.join(vis_dir, frame_name.split('.')[0] + '.jpg'),bbox_inches='tight')
                    plt.close(fig)


            saved_dir_path = os.path.join(saved_dir, "terminal_image_frames",area,date)
            if not os.path.exists(saved_dir_path):
                    os.makedirs(saved_dir_path)
            with open(os.path.join(saved_dir_path, f"{date}.json"), 'w', encoding='utf-8') as js:
                # 使用json.dump序列化list并存储到文件
                json.dump(area_timestamp_dict, js, ensure_ascii=False, indent=4)



                




    # for image_dir_path in image_dir_paths:
    #     for frame_name in os.listdir(image_dir_path):
    #         frame_path = os.path.join(image_dir_path, frame_name)
    #         pred_dmap, pred_count =predict(model, frame_path, config_mp["device"], config_mp["unit_size"],config_mp["patch_size"], config_mp["log_para"])




                
    
    
    
    

# def load_imgs(img_path, unit_size, device):
#     if os.path.isdir(img_path):
#         img_paths = glob(os.path.join(img_path, '*'))
#     else:
#         img_paths = [img_path]

#     imgs = []
#     for img_path in img_paths:
#         assert os.path.exists(img_path), f'Image path {img_path} does not exist.'
#         assert img_path.lower().endswith('.jpg') or img_path.lower().endswith('.jpeg') \
#             or img_path.lower().endswith('.png'), 'Only support .jpg and .png image format.'
        
#         img = Image.open(img_path).convert('RGB')
        
#         if unit_size > 0:
#             w, h = img.size
#             new_w = (w // unit_size + 1) * unit_size if w % unit_size != 0 else w
#             new_h = (h // unit_size + 1) * unit_size if h % unit_size != 0 else h

#             padding, h, w = get_padding(h, w, new_h, new_w)

#             img = F.pad(img, padding)
#         img = F.to_tensor(img)
#         img = F.normalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#         img = img.unsqueeze(0).to(device)
#         imgs.append(img)

#     img_names = [os.path.basename(img_path) for img_path in img_paths]

#     return imgs, img_names




# def inferenceMP(config):
#     imgs, img_names = load_imgs(config["img_path"], config["unit_size"], config["device"])
#     model = load_model(config["pretrained"], config["device"])
#     return model
    

# def main(args):
#     imgs, img_names = load_imgs(args.img_path, args.unit_size, args.device)
#     model = load_model(args.model_path, args.device)

#     start_time = time()
#     for img, img_name in zip(imgs, img_names):
#         pred_dmap, pred_count = predict(model, img, args.patch_size, args.log_para)
#         print(f'{img_name}: {pred_count}')

#         if args.save_path is not None:
#             with open(args.save_path, 'a') as f:
#                 f.write(f'{img_name}: {pred_count}\n')

#         if args.vis_dir is not None:
#             os.makedirs(args.vis_dir, exist_ok=True)
#             denormed_img = denormalize(img)[0].cpu().permute(1, 2, 0).numpy()
#             fig = plt.figure(figsize=(10, 5))
#             ax_img = fig.add_subplot(121)
#             ax_img.imshow(denormed_img)
#             ax_img.set_title(img_name)
#             ax_dmap = fig.add_subplot(122)
#             ax_dmap.imshow(pred_dmap)
#             ax_dmap.set_title(f'Predicted count: {pred_count}')
#             import matplotlib
#             matplotlib.image.imsave(os.path.join(args.vis_dir,f"{img_name[:-4]}({int(pred_count)}).jpg"), pred_dmap)
#             plt.savefig(os.path.join(args.vis_dir, img_name.split('.')[0] + '.png'))
#             plt.close(fig)
#     print(f'Total time: {time()-start_time:.2f}s')

# if __name__ == '__main__':
#     import yaml
#     with open('recog_model/vConfig.yaml','r') as f:
#         vconfig = yaml.load(f,Loader=yaml.Loader)
#         print(vconfig)
#     parser = ArgumentParser()
#     parser.add_argument('--img_path', default=vconfig["img_path"],type=str, help='Path to the image or directory containing images.')
#     parser.add_argument('--model_path',default= vconfig['model_path'], type=str, help='Path to the model weight.')
#     parser.add_argument('--save_path', type=str, default=vconfig['save_path'], help='Path of the text file to save the prediction results.')
#     parser.add_argument('--vis_dir', type=str, default=vconfig['vis_dir'], help='Directory to save the visualization results.')
#     parser.add_argument('--unit_size', type=int, default=16, help='Unit size for image resizing. Normally set to 16 and no need to change.')
#     parser.add_argument('--patch_size', type=int, default=3584, help='Patch size for image division. Decrease this value if OOM occurs.')
#     parser.add_argument('--log_para', type=int, default=1000, help='Parameter for log transformation. Normally set to 1000 and no need to change.')
#     parser.add_argument('--device', type=str, default='cuda', help='Device to run the model. Default is cuda.')
#     args = parser.parse_args()

#     main(args)
    
    
    
# python inference.py --img_path my_imgs --model_path sta.pth --save_path output.txt --vis_dir vis
