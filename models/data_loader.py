
class CustomDataLoader(Sequence):
    def __init__(self, dataset_dir, batch_size=32, mode='train', shuffle=True):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.mode = mode
        self.shuffle = shuffle
        self.image_dir = os.path.join(dataset_dir, mode, 'image')
        self.mask_dir = os.path.join(dataset_dir, mode, 'mask')
    # Filter out non-image files from the list of image and mask files
        self.image_files = [f for f in sorted(os.listdir(self.image_dir)) if f.endswith('.npy')]
        self.mask_files = [f for f in sorted(os.listdir(self.mask_dir)) if f.endswith('.npy')]
    
        self.indexes = np.arange(len(self.image_files))
        if shuffle:
            np.random.shuffle(self.indexes)

        # Check if image and mask files are correctly paired
        for image_file, mask_file in zip(self.image_files, self.mask_files):
            image_number = image_file.split('_')[1].split('.')[0]
            mask_number = mask_file.split('_')[1].split('.')[0]
            if image_number != mask_number:
                raise ValueError(f"Mismatched image and mask files: {image_file} and {mask_file}")

    def __len__(self):
        return int(np.ceil(len(self.image_files) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        if end_idx > len(self.indexes):
            end_idx = len(self.indexes)

        batch_indexes = self.indexes[start_idx:end_idx]
        batch_images = []
        batch_masks = []

        for idx in batch_indexes:
            image_filename = os.path.join(self.image_dir, self.image_files[idx])
            mask_filename = os.path.join(self.mask_dir, self.mask_files[idx])

            image = np.load(image_filename)
            mask = np.load(mask_filename)

            batch_images.append(image)
            batch_masks.append(mask)

        batch_images = np.array(batch_images)
        batch_masks = np.array(batch_masks)

        return batch_images, batch_masks

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)