# This script filters COCO 2017 dataset for target classes and converts annotations to YOLO format
# Usage:
# python download_coco_subset.py --coco_annotations path/to/instances_val2017.json --images_dir path/to/val2017

import os, argparse, shutil
from pycocotools.coco import COCO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_annotations', required=True)
    parser.add_argument('--images_dir', required=True)
    parser.add_argument('--out_dir', default='../datasets/real_coco_subset')
    parser.add_argument('--classes', nargs='+', default=['cup','bottle','chair','laptop','book'])
    args = parser.parse_args()

    coco = COCO(args.coco_annotations)
    cat_ids = coco.getCatIds(catNms=args.classes)
    img_ids = set()
    for cid in cat_ids:
        img_ids.update([ann['image_id'] for ann in coco.loadAnns(coco.getAnnIds(catIds=[cid]))])

    os.makedirs(os.path.join(args.out_dir,'images'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir,'labels'), exist_ok=True)

    for img_id in list(img_ids)[:250]:
        info = coco.loadImgs([img_id])[0]
        fname = info['file_name']
        shutil.copy2(os.path.join(args.images_dir, fname), os.path.join(args.out_dir,'images', fname))
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id], catIds=cat_ids))
        label_lines = []
        for a in anns:
            x,y,w,h = a['bbox']
            x_c = (x + w/2)/info['width']
            y_c = (y + h/2)/info['height']
            w_n = w/info['width']
            h_n = h/info['height']
            cat_idx = args.classes.index(coco.loadCats([a['category_id']])[0]['name'])
            label_lines.append(f"{cat_idx} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")
        with open(os.path.join(args.out_dir,'labels', os.path.splitext(fname)[0]+'.txt'),'w') as f:
            f.writelines(label_lines)

if __name__ == '__main__':
    main()
