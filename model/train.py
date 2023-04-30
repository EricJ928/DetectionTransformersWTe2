import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from tqdm import tqdm
from PIL import Image, ImageDraw
from transformers import DetrFeatureExtractor, DetrForObjectDetection


# Training dataset preprocessing using TorchVision CocoDetection dataset.
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, feature_extractor, train=True):
        ann_file = os.path.join(img_folder, "train_coco.json" if train else "val_coco.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)
        
        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target


'''Next, let's create corresponding dataloaders. We define a custom collate_fn to batch images together.
As DETR resizes images to have a min size of 800 and a max size of 1333, images can have different sizes. 
We pad images (pixel_values) to the largest image in a batch, and create a corresponding pixel_mask to 
indicate which pixels are real (1)/which are padding (0).'''
def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = labels
    return batch


# Model
class Detr(nn.Module):
    def __init__(self):
        super(Detr, self).__init__()
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    def forward(self, pixel_values, pixel_mask, labels):
        out = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        return out


if __name__ == '__main__':
    # train_img_folder = 'WTe2_recognition/data/train/'
    # val_img_folder = 'WTe2_recognition/data/train/'
    train_img_folder = 'WTe2_recognition/WTe2_examples/all/png/good/object_detection/WTe2/train'
    val_img_folder = 'WTe2_recognition/WTe2_examples/all/png/good/object_detection/WTe2/val'
    save_folder = 'WTe2_recognition/WTe2_examples/all/png/good/object_detection/WTe2'

    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

    train_dataset = CocoDetection(img_folder=train_img_folder, feature_extractor=feature_extractor)
    val_dataset = CocoDetection(img_folder=val_img_folder, feature_extractor=feature_extractor, train=False)
    print('-'*100)
    print('Number of training samples:', len(train_dataset))
    print('Number of validation samples:', len(val_dataset))

    # based on https://github.com/woctezuma/finetune-detr/blob/master/finetune_detr.ipynb
    image_ids = train_dataset.coco.getImgIds()
    # let's pick a random image
    image_id = image_ids[np.random.randint(0, len(image_ids))]
    print('Image #{}'.format(image_id))
    image = train_dataset.coco.loadImgs(image_id)[0]
    image = Image.open(os.path.join(train_img_folder, image['file_name']))

    annotations = train_dataset.coco.imgToAnns[image_id]
    draw = ImageDraw.Draw(image, "RGBA")

    cats = train_dataset.coco.cats
    print(cats)
    id2label = {k: v['name'] for k,v in cats.items()}

    # print(annotations)
    # for annotation in annotations:
    #     box = annotation['bbox']
    #     class_idx = annotation['category_id']
    #     x,y,w,h = tuple(box)
    #     draw.rectangle((x,y,x+w,y+h), outline='#ff145e' if class_idx==2 else '#00ffff', width=1)
    #     draw.text((x, y), id2label[class_idx], fill='white')

    # image.show()

    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=4, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=2)
    batch = next(iter(train_dataloader))
    print('-'*100)
    print('batch key:', batch.keys())
    pixel_values, target = train_dataset[0]
    print('pixel_values:', pixel_values.shape)


    # Train Dataset
    print('-'*100)
    print('Start training...')

    # Instantiate pre-trained DETR model with randomly initialized classification head
    # model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    model = Detr()

    # I almost always use a learning rate of 5e-5 when fine-tuning Transformer based models
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    # put model on GPU, if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    model.to(device)

    epochs = 50
    log = {'epoch': [], 'loss': []}
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        model.train()
        train_loss = 0.0
        for batch in train_dataloader:
            # put batch on device
            pixel_values = batch["pixel_values"]
            pixel_mask = batch["pixel_mask"]
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
            
            # forward pass
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss = outputs.loss
            
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        log['epoch'].append(epoch)
        log['loss'].append(train_loss/len(train_dataloader))
        # print("Loss after epoch {epoch}:", train_loss/len(train_dataloader))
        pbar.set_postfix({'Loss': train_loss/len(train_dataloader)})
    
    print('-'*100)
    print('Training: done')
    torch.save(model.state_dict(), f'{save_folder}/epochs_{epochs}.pt')
    
    data_df = pd.DataFrame(log['epoch'], columns=['epoch'])
    data_df['loss'] = log['loss']
    data_df.to_csv(f'{save_folder}/epochs_{epochs}.csv', index=False)
