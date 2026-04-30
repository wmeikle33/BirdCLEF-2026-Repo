PATH = '/kaggle/input/competitions/birdclef-2026/'
TEST_PATH = PATH + 'test_soundscapes/'
TRAIN_PATH = PATH + 'train_soundscapes/'

class BirdDataset(Dataset):
    taxonomy = pd.read_csv(PATH+'taxonomy.csv')
    
    LABELS = list(np.unique(taxonomy.primary_label))
    CLASSES = list(np.unique(taxonomy.class_name))
    BATCH_SIZE = 32

    DUR = 5
    SR = 32000

    def __init__(self, split_size=0.2, seed=2, n_repeat=1, is_train=True):
        paths = [self.TEST_PATH+x for x in os.listdir(self.TEST_PATH) if '.ogg' in x]
        if len(paths)==0:
            paths = [self.TRAIN_PATH+x for x in os.listdir(self.TRAIN_PATH) if '.ogg' in x]
            paths = sorted(paths)[:16]
        df = pd.DataFrame([], index=range(len(paths)*int(60/self.DUR)), columns=['path', 'start', 'end'])
        self.paths = paths.copy()


    def __len__(self):
        return len(self.paths)
