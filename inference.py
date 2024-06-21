if __name__ == '__main__':
    import argparse
    import numpy as np
    from torch.utils.data import DataLoader
    import torch
    import torchvision.transforms.functional as TF
    import logging
    import os
    import matplotlib.pyplot as plt
    from data.datasets import ManipulationDataset
    from models.cmnext_conf import CMNeXtWithConf
    from models.modal_extract import ModalitiesExtractor
    from configs.cmnext_init_cfg import _C as config, update_config
    import csv

    parser = argparse.ArgumentParser(description='Infer')
    parser.add_argument('-gpu', '--gpu', type=int, default=0, help='device, use -1 for cpu')
    parser.add_argument('-log', '--log', type=str, default='INFO', help='logging level')
    parser.add_argument('-exp', '--exp', type=str, default='experiments/ec_example_phase2.yaml', help='Yaml experiment file')
    parser.add_argument('-ckpt', '--ckpt', type=str, default='ckpt/early_fusion_detection.pth', help='Checkpoint')
    parser.add_argument('-path', '--path', type=str, default='example_folder/', help='Folder path containing images')
    parser.add_argument('opts', help="other options", default=None, nargs=argparse.REMAINDER)
    parser.add_argument('-score_file','--score_file', type=str, help='score file')
    args = parser.parse_args()

    score_file = args.score_file
    csv_file = "scores"+score_file+"_MM.csv"


    config = update_config(config, args.exp)

    loglvl = getattr(logging, args.log.upper())
    logging.basicConfig(level=loglvl)

    gpu = args.gpu

    device = 'cuda:%d' % gpu if gpu >= 0 else 'cpu'
    device='cpu'
    np.set_printoptions(formatter={'float': '{: 7.3f}'.format})

    if device != 'cpu':
        # cudnn setting
        import torch.backends.cudnn as cudnn

        cudnn.benchmark = False
        cudnn.deterministic = True
        cudnn.enabled = config.CUDNN.ENABLED


    modal_extractor = ModalitiesExtractor(config.MODEL.MODALS[1:], config.MODEL.NP_WEIGHTS)

    model = CMNeXtWithConf(config.MODEL)

    # ckpt = torch.load(args.ckpt)
    ckpt = torch.load(args.ckpt, map_location=torch.device('cpu'))


    model.load_state_dict(ckpt['state_dict'])
    modal_extractor.load_state_dict(ckpt['extractor_state_dict'])

    modal_extractor.to(device)
    model = model.to(device)
    modal_extractor.eval()
    model.eval()
    image_files = [f for f in os.listdir(args.path) if os.path.isfile(os.path.join(args.path, f))]
    file_list_path = 'tmp_inf.txt'
    with open(file_list_path, 'w') as f:
        for image_file in image_files:
            f.write(os.path.join(args.path, image_file) + ' None 0\n')


    val = ManipulationDataset('tmp_inf.txt',
                            config.DATASET.IMG_SIZE,
                            train=False)
    val_loader = DataLoader(val,
                            batch_size=1,
                            shuffle=False,
                            num_workers=min(config.WORKERS, 8), 
                            pin_memory=True)

    file_scores = []



    for step, (images, _, masks, lab,filenames) in enumerate(val_loader):
        target = "./data/mask/" + os.path.basename(filenames[0]).split(".")[0] + "_mask.png"
        
        with torch.no_grad():
            
            images = images.to(device, non_blocking=True)
            masks = masks.squeeze(1).to(device, non_blocking=True)

            modals = modal_extractor(images)

            images_norm = TF.normalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            inp = [images_norm] + modals

            anomaly, confidence, detection = model(inp)

            gt = masks.squeeze().cpu().numpy()
            map = torch.nn.functional.softmax(anomaly, dim=1)[:, 1, :, :].squeeze().cpu().numpy()
            det = detection.item()

            target = "./data/mask/" + os.path.basename(filenames[0]).split(".")[0] + "_mask.png"
            
            file_scores.append([os.path.basename(filenames[0]).split(".")[0], det])
        
            if os.path.exists(target):
                print(f"Mask already exists for {filenames[0]}, skipping.")
                continue
            plt.imsave(target, map, cmap='RdBu_r', vmin=0, vmax=1)
            print(f"Processed {filenames[0]}")
            print(f"Detection score: {det}")
            print(f"Localization map saved in {target}")


    print(f"Processed all images in {args.path}")

    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['File Name', 'Score'])
        writer.writerows(file_scores)

    print(f"Scores saved to {csv_file}")
