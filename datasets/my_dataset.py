import os
import pickle

def my_dataset_func(data_dir, phase):
    """dataset func for detectron2.
       ref: https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#standard-dataset-dicts
       params:
          data_dir : "path/to/datadir"
          phase : "train" or "val"
       Returns:
          function which returns list of dict. Each dict contains information about one image.
    """
    VAL_IMAGE_IDS = [
        "0c2550a23b8a0f29a7575de8c61690d3c31bc897dd5ba66caec201d201a278c2",
        "92f31f591929a30e4309ab75185c96ff4314ce0a7ead2ed2c2171897ad1da0c7",
        "1e488c42eb1a54a3e8412b1f12cde530f950f238d71078f2ede6a85a02168e1f",
        "c901794d1a421d52e5734500c0a2a8ca84651fb93b19cec2f411855e70cae339",
        "8e507d58f4c27cd2a82bee79fe27b069befd62a46fdaed20970a95a2ba819c7b",
        "60cb718759bff13f81c4055a7679e81326f78b6a193a2d856546097c949b20ff",
        "da5f98f2b8a64eee735a398de48ed42cd31bf17a6063db46a9e0783ac13cd844",
        "9ebcfaf2322932d464f15b5662cae4d669b2d785b8299556d73fffcae8365d32",
        "1b44d22643830cd4f23c9deadb0bd499fb392fb2cd9526d81547d93077d983df",
        "97126a9791f0c1176e4563ad679a301dac27c59011f579e808bbd6e9f4cd1034",
        "e81c758e1ca177b0942ecad62cf8d321ffc315376135bcbed3df932a6e5b40c0",
        "f29fd9c52e04403cd2c7d43b6fe2479292e53b2f61969d25256d2d2aca7c6a81",
        "0ea221716cf13710214dcd331a61cea48308c3940df1d28cfc7fd817c83714e1",
        "3ab9cab6212fabd723a2c5a1949c2ded19980398b56e6080978e796f45cbbc90",
        "ebc18868864ad075548cc1784f4f9a237bb98335f9645ee727dac8332a3e3716",
        "bb61fc17daf8bdd4e16fdcf50137a8d7762bec486ede9249d92e511fcb693676",
        "e1bcb583985325d0ef5f3ef52957d0371c96d4af767b13e48102bca9d5351a9b",
        "947c0d94c8213ac7aaa41c4efc95d854246550298259cf1bb489654d0e969050",
        "cbca32daaae36a872a11da4eaff65d1068ff3f154eedc9d3fc0c214a4e5d32bd",
        "f4c4db3df4ff0de90f44b027fc2e28c16bf7e5c75ea75b0a9762bbb7ac86e7a3",
        "4193474b2f1c72f735b13633b219d9cabdd43c21d9c2bb4dfc4809f104ba4c06",
        "f73e37957c74f554be132986f38b6f1d75339f636dfe2b681a0cf3f88d2733af",
        "a4c44fc5f5bf213e2be6091ccaed49d8bf039d78f6fbd9c4d7b7428cfcb2eda4",
        "cab4875269f44a701c5e58190a1d2f6fcb577ea79d842522dcab20ccb39b7ad2",
        "8ecdb93582b2d5270457b36651b62776256ade3aaa2d7432ae65c14f07432d49",
    ]
 
    def inner_func():
        preprocessed = os.path.join(data_dir, f"detectron2_format_{phase}.pkl")
        if os.path.exists(preprocessed):
            print("load cached dataset..")
            with open(preprocessed, mode="rb") as f:
                ret = pickle.load(f) # cache
                return ret
 
        print(f"creating {phase} dataset..")
        ret = []
        for img_id in tqdm(os.listdir(data_dir)):
 
            if phase == "train":
                if img_id in VAL_IMAGE_IDS:
                    continue
            elif phase == "val":
                if img_id not in VAL_IMAGE_IDS:
                    continue
 
            if os.path.isfile(img_id):
                continue
 
            img_path = os.path.join(data_dir, img_id, "images", img_id + ".png")
            img = cv2.imread(img_path)
 
            if img is None:
                continue
 
            masks_path = [str(p) for p in Path(os.path.join(data_dir, img_id, "masks")).glob("*.png")]
            annotations = []
            for p in masks_path:
                mask = cv2.imread(p) # (0,255)
                mask_bin = np.all(mask == 255, axis=2).astype(np.uint8) # (0,1)
                segmentation = pycocotools.mask.encode(np.asarray(mask_bin, order="F"))
                mask_indexes = np.where(mask_bin)
                y_min = np.min(mask_indexes[0])
                y_max = np.max(mask_indexes[0])
                x_min = np.min(mask_indexes[1])
                x_max = np.max(mask_indexes[1])
                annotations.append({
                    "bbox" : [x_min, y_min, x_max - x_min, y_max - y_min],
                    "bbox_mode" : BoxMode.XYWH_ABS,
                    #"bbox" : [x_min, y_min, x_max, y_max],
                    #"bbox_mode" : BoxMode.XYXY_ABS
                    "category_id" : 0, # nucleus
                    "segmentation" : segmentation,
                })
            ret.append({
                "file_name": img_path,
                "height"   : img.shape[0],
                "width"    : img.shape[1],
                "image_id" : img_id,
                "annotations" :annotations , # Each dict corresponds to annotations of one instance in this image,
            })
 
        with open(preprocessed, mode="wb") as f:
            pickle.dump(ret, f) # cache
 
        return ret
 
    return inner_func

