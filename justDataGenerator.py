class DataGenerator(Sequence):
    def __init__(self, Data, age, gender, ethnicity, dataDir, num_classes, batch_size = 32, shuffle=False):
        '''
        Data: List of images
        age: Age corresponding to the Data (Regression: Continuous Value)
        gender, ethnicity: list of gender and ethnicity as labels
        dataDir: Complete path to Data
        num_classes: list of classes in gender and ethnicit
        batch_size: Number of data points to feed
        '''
        self.Data = Data
        self.age = age
        self.gender = gender
        self.ethnicity = ethnicity
        self.dataDir = dataDir
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def on_epoch_end(self):
        ## update indexes after epoch end
        self.indexes = np.arange(len(self.Data))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)
        
    def __len__(self):
        ## Number of data in batch per epoch
        return int(np.floor(len(self.Data)/self.batch_size))
    
    def __getitem__(self, index):
        ## Return X,y for each batch index
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        Data_temp = [self.Data[i] for i in indexes]
        X, y = self._DataGeneration(Data_temp)
        
        return (X,y)
    
    def _DataGeneration(self, Data_temp):
        X = []
        y = []
        sizes = []

        H,W,C = 400,400,3
        
        for data in Data_temp:
            img = cv2.imread(self.dataDir+'/'+data)
            img = cv2.resize(img, (H,W)) 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.resize is True:
                img = cv2.resize(img, self.resized_dim)
            img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            X.append(img)
            y.append(self.labels[data])
            
        return np.array(X).reshape(self.batch_size,H,W,3), to_categorical(np.array(y), self.num_classes, dtype='float32')