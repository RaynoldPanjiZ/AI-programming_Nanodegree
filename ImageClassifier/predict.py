import json
import argparse
from time import time
from PIL import Image
import numpy as np

import torch


parser = argparse.ArgumentParser()
parser.add_argument('img_dir', type = str,
        help = 'path to the file image: /path/to/image') 
parser.add_argument('checkpoint',
        help = 'load from saved model: /path/to/model_checkpoint.pth')
parser.add_argument('--top_k', default = 1, type = int)
parser.add_argument('--category_names', default = 'cat_to_name.json',
        help = 'categorical label encoding to actual name') 
parser.add_argument('--gpu', action="store_true", default=False )
in_args = parser.parse_args()

def main():
    img_tensor = process_image(in_args.img_dir)

    model, idx_to_class, _ = load_checkpoint(in_args.checkpoint)

    if in_args == True:
        if torch.cuda.is_available() == False:
            img_tensor.to("cpu")
            model.to("cpu")
            print("GPU not found!!")
        else:
            img_tensor.to("cuda:0")
            model.to("cuda:0")
    else:
        img_tensor.to("cpu")
        model.to("cpu")

    if (in_args.top_k <= 0):
        print(" TOP_K number cannot be less than or equal to 0 !!")
        exit()
        
    start_time = time()  
    probs, classes = predict(img_tensor, model, topk=in_args.top_k)
    tot_time = time() - start_time

    display_predict(probs, classes, idx_to_class, tot_time)
    

def display_predict(probs, classes, idx_to_class, tot_time):
    with open(in_args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    idx_labels = [idx_to_class[y] for y in classes]
    labels = [cat_to_name[str(y)] for y in idx_labels]
    
    print(f"top {in_args.top_k} classification result:")
    for i, (prob, lb) in enumerate(zip(probs, labels)):
        print("{}. classification result: [{}], probabilities: {}".format(i+1, lb, str(prob)))
    
    print()
    print(f"total time: {tot_time:.2f} s")
    
    
def process_image(image_path):
    img = Image.open(image_path)
    width, height = img.size  

    if width < height:
        width, height = (255, int(255*(height/width)))
    else:
        width, height = (int(255*(width/height)), 255)

    img = img.resize((width, height)) 
    width_r, height_r = img.size  
    left = (width_r - 224)/2
    top = (height_r - 224)/2
    right = (width_r + 224)/2
    bottom = (height_r + 224)/2
    img = img.crop((left, top, right, bottom)) 

    img = np.array(img) 
    img = img.transpose((2, 0, 1))  
    img = img/255

    img[0] = (img[0] - 0.485)/0.229
    img[1] = (img[1] - 0.456)/0.224
    img[2] = (img[2] - 0.406)/0.225

    img = img[np.newaxis,:] 
    image = torch.from_numpy(img)
    image = image.float()
    return image


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    epoch = checkpoint['epochs']
    idx_to_class = checkpoint['idx_to_class']
    acc = checkpoint['val_acc']

    model.eval()
    return model, idx_to_class, acc


def predict(image, model, topk):
    with torch.no_grad():
        outputs = model(image)
        probs, classes = torch.exp(outputs).topk(topk)
        return probs[0].tolist(), classes[0].add(1).tolist()


if __name__ == "__main__":
    main()