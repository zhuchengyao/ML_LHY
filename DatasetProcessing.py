# rewrite the Dataset class
class FoodDataset(Dataset):
    def __init__(self, path, tfm=test_tfm, files = None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        # listdir return everyfile ended with .jpg into x, and os.path.join combine the directory of jpgs.
        if files != None:
            self.files = files
        self.transform = tfmx.endswish(".jpg")

    def __len__(self):
        return self.len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        try:
            label = int(fname.split("/")[-1].split("_")[0]) # first split mean use "/" as a sign to split the
            # whole string, the string has been split into 2 parts.[-1] means select the last part.
        except:
            label = -1  # test has no label
        return im, label

