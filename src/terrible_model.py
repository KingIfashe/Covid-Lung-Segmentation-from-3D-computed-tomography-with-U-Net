## code that gives black screen predcitions / bad predctions 

import tensorflow as tf
import numpy as np
import SimpleITK as sitk
import glob
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU  # You might need to implement Dice coefficient yourself
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

#def augment(image, mask):
   # """Apply data augmentation to images and masks."""
    #image = tf.image.random_flip_left_right(image)
    #mask = tf.image.random_flip_left_right(mask)
    # Add more augmentation transformations if needed
    #return image, mask

def dice_coefficient(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def unet_model(input_size=(128, 128, 1)):
    inputs = Input(input_size)
    # Contracting Path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    # Expanding Path
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    
    # Output Layer
    # Note: Use 'sigmoid' for binary-class segmentation
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)  # Assuming 9 classes: background, left lung, right lung, infection
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

model  = unet_model()


# Step 2: Prepare the Dataset

def load_nifti_img(filepath, dtype=np.float32):
    """Load a NIfTI file as a numpy array."""
    sitk_img = sitk.ReadImage(filepath)
    img_array = sitk.GetArrayFromImage(sitk_img)
    return img_array.astype(dtype)
    
def resize_img_3d(img, target_shape, is_mask=False):
    # Calculate the resize ratios for each dimension
    resize_ratios = [target_shape[i] / img.shape[i] for i in range(3)]
    
    # Apply the resize ratios
    resized_img = zoom(img, resize_ratios, order=0 if is_mask else 3)  # Nearest for masks, cubic for images
    
    # Calculate necessary padding for each spatial dimension
    padding = [(0, target_shape[i] - resized_img.shape[i]) for i in range(3)]
    
    # Correctly apply padding
    # If you plan to add a channel dimension later, no additional padding is needed here for the channel
    padded_img = np.pad(resized_img, padding + [(0, 0)] * (4 - len(padding)), mode='constant', constant_values=0)
    
    return padded_img

# Pattern to match all .nii files within the specified directories
image_pattern = '/Users/kingifashe/medical_imaging/COVID-19-CT-Seg_20cases/*.nii'
mask_pattern = '/Users/kingifashe/medical_imaging/Infection_Mask/*.nii'
# Use glob to find all files matching the pattern
image_paths = glob.glob(image_pattern)
mask_paths = glob.glob(mask_pattern)

# Sort the paths to ensure corresponding images and masks align,
# assuming their filenames are consistent and sortable for alignment
image_paths.sort()
mask_paths.sort()

# Define split ratios
train_ratio = 0.7
val_test_ratio = 0.3
val_ratio_relative = 0.5  # Relative to val_test_ratio

# Calculate split indices
num_images = len(image_paths)
num_train = int(num_images * train_ratio)
num_val_test = num_images - num_train
num_val = int(val_test_ratio * val_ratio_relative * num_images)

# Split the paths into training, validation, and test sets
train_image_paths = image_paths[:num_train]
train_mask_paths = mask_paths[:num_train]

val_image_paths = image_paths[num_train:num_train + num_val]
val_mask_paths = mask_paths[num_train:num_train + num_val]

test_image_paths = image_paths[num_train + num_val:]
test_mask_paths = mask_paths[num_train + num_val:]

print("Found image files:", image_paths)
print("Found mask files:", mask_paths)



#def center_crop(arr, new_size):
    #"""
    #Center crops a square numpy array to a specified size.
    
   # Parameters:
    #- arr: Input 2D square numpy array.
   # - new_size: The size of the sides of the square crop.
    
    #Returns:
    #- Cropped 2D numpy array of size (new_size, new_size).
    #"""
    ## Ensure the new size is not larger than the array dimensions
    #if new_size > arr.shape[0]:
        #print(arr.shape[0])
    #assert new_size <= arr.shape[0] and new_size <= arr.shape[1], "new_size is too large"
    
    # Calculate center, start, and end indices
    #center_y, center_x = arr.shape[0] // 2, arr.shape[1] // 2
    #start_y = center_y - new_size // 2
    #start_x = center_x - new_size // 2
    #end_y = start_y + new_size
    #end_x = start_x + new_size
    
    # Crop the array
    #cropped_arr = arr[start_y:end_y, start_x:end_x]
    
    #return cropped_arr



def resize_or_pad_slice(img_2d, target_shape=(128, 128), is_mask=False):
    """
    Resize or pad a 2D slice to a target shape and add a channel dimension.
    
    Args:
        img_2d (numpy.ndarray): The input 2D slice as a numpy array.
        target_shape (tuple): The desired output size as (height, width).
        is_mask (bool): Whether the input slice is a mask, affecting interpolation method.
        
    Returns:
        numpy.ndarray: The resized (and potentially padded) slice as a numpy array with a channel dimension.
    """
    # Ensure img_2d is a 3D tensor with the single channel dimension
    if img_2d.ndim == 2:
        img_2d = np.expand_dims(img_2d, axis=-1)

    # Convert numpy array to tensor
    img_tensor = tf.convert_to_tensor(img_2d, dtype=tf.float32)

    resized_img = tf.image.resize_with_pad(
        img_tensor, 
        target_shape[0], 
        target_shape[1], 
        method=('nearest' if is_mask else 'bilinear')
    )
    # The output is already a 3D tensor including the channel dimension,
    # so no further channel dimension adjustment is needed.
    return resized_img.numpy()

def load_and_preprocess2d(image_paths, mask_paths, target_shape=(128, 128)):
    images = []
    masks = []
    
    for img_path, mask_path in zip(image_paths, mask_paths):
        img_3d = load_nifti_img(img_path)
        mask_3d = load_nifti_img(mask_path, dtype=np.uint8)
        
        for i in range(img_3d.shape[0]):  # Iterate through slices
            img_2d = img_3d[i, :, :]
            mask_2d = mask_3d[i, :, :]
            
            # Resize or pad the 2D slices and ensure they have a channel dimension
            img_2d_resized = resize_or_pad_slice(img_2d, target_shape)
            mask_2d_resized = resize_or_pad_slice(mask_2d, target_shape, is_mask=True)

           # if img_2d.shape[0] >=512:
              #  img_2d_resized = center_crop(img_2d, 512)
              #  mask_2d_resized = center_crop(img_2d, 512)
            # Debug: Print shape of each slice after resizing/padding
            print("Resized image slice shape:", img_2d_resized.shape, "Resized mask slice shape:", mask_2d_resized.shape) #[512,512] or [630,630]

            images.append(img_2d_resized)
            masks.append(mask_2d_resized)
    
    # Convert lists to numpy arrays without immediate reshaping to debug
    print(len(images))
    print(len(images[-1]))
    images = np.array(images)
    masks = np.array(masks)
    
    # Debug: Print overall shapes before reshaping
    print("Overall images shape before reshape:", images.shape)
    print("Overall masks shape before reshape:", masks.shape)

    # Reshape if all shapes are correct
    images = images.reshape((-1, *target_shape, 1))
    masks = masks.reshape((-1, *target_shape, 1))
    
    return images, masks

# Using the function to load and preprocess datasets
train_images, train_masks = load_and_preprocess2d(train_image_paths, train_mask_paths, target_shape=(128, 128))
val_images, val_masks = load_and_preprocess2d(val_image_paths, val_mask_paths, target_shape=(128, 128))
test_images, test_masks = load_and_preprocess2d(test_image_paths, test_mask_paths, target_shape=(128, 128))



class TFDataGenerator(tf.data.Dataset):
    def __new__(cls, image_paths, mask_paths, target_shape=(128, 128)):
        """
        Create a TensorFlow data generator for loading and preprocessing 2D slices from 3D CT scans.

        Args:
            image_paths (list): List of paths to the image files.
            mask_paths (list): List of paths to the mask files.
            target_shape (tuple): Target resize shape for each slice.

        Returns:
            A `tf.data.Dataset` object.
        """
        cls.image_paths = image_paths
        cls.mask_paths = mask_paths
        cls.target_shape = target_shape

        def generator():
            for img_path, mask_path in zip(cls.image_paths, cls.mask_paths):
                # Load the 3D image and mask using a predefined function
                img_3d = load_nifti_img(img_path, dtype=np.float32)
                mask_3d = load_nifti_img(mask_path, dtype=np.uint8)
                
                for slice_idx in range(img_3d.shape[0]):
                    img_2d = img_3d[slice_idx, :, :]
                    mask_2d = mask_3d[slice_idx, :, :]
                    
                    # Resize or pad the 2D slices
                    img_2d_resized, mask_2d_resized = resize_or_pad_slice(img_2d, cls.target_shape), resize_or_pad_slice(mask_2d, cls.target_shape, is_mask=True)

                    # Normalize the image
                    img_2d_resized = img_2d_resized / 255.0

                    # Ensure mask is in the correct format
                    mask_2d_resized = mask_2d_resized / 255.0
                    mask_2d_resized = np.expand_dims(mask_2d_resized, axis=-1)  # Add channel dimension for consistency with img
                    
                    yield img_2d_resized, mask_2d_resized

        # Define the output types and shapes
        output_types = (tf.float32, tf.float32)
        output_shapes = (tf.TensorShape([*target_shape, 1]), tf.TensorShape([*target_shape, 1]))

        return tf.data.Dataset.from_generator(
            generator,
            output_types=output_types,
            output_shapes=output_shapes
        )

# Convert the numpy arrays into TensorFlow datasets

batch_size = 8

# Batch the datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks)).batch(batch_size).shuffle(len(train_images))
#train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks)).map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).shuffle(len(train_images))

val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_masks)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_masks)).batch(batch_size)

print(f'Number of training batches: {len(train_dataset)}')
print(f'Number of validation batches: {len(val_dataset)}')
print(f'Number of test batches: {len(test_dataset)}')

for images, masks in train_dataset.take(4):
    print("Image batch shape:", images.shape)
    print("Mask batch shape:", masks.shape)
    # Expect: Image batch shape: (8, 128, 128, 1), Mask batch shape: (8, 128, 128, 1)


#def focal_loss(gamma=2., alpha=0.25):
    #def focal_loss_fixed(y_true, y_pred):
        #pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        #pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        #return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    #return focal_loss_fixed

# Compile the model
model  = unet_model()
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])#focal loss?
#loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, from_logits=False)
#loss=focal_loss(gamma=2., alpha=0.25)
model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy', dice_coefficient])
callbacks_list = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
]
# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)  # Adjust the number of epochs as needed
#model.fit(train_dataset_augmented, validation_data=val_dataset, epochs=10, callbacks=[...])



# Save the entire model as a `.keras` zip archive.
model.save('my_model.keras')

# Evaluate the model on the test set
test_loss, test_acc, test_dice = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}, Test Dice Coefficient: {test_dice}")


# Make predictions
predictions_all = []

for test_images, test_masks in test_dataset.take(4):  # Take a batch from the test dataset
    predictions = model.predict(test_images)
    predictions_binary = (predictions > 0.5).astype(np.float32)
    predictions_all.append(predictions_binary) # Apply threshold to convert probabilities to binary mask

# Now concatenate all the predictions
predictions_concatenated = np.concatenate(predictions_all, axis=0)
    
    # Visualization code here
    # You can use matplotlib to visualize the original images, true masks, and predicted masks
#print(np.concatenate(predictions_all, axis=0).shape)
print(predictions.shape)

 #Now, you can visualize the original images, true masks, and predicted (thresholded) masks
#num_examples = min(len(test_images), 8)  # Choose a number that you have in your batch, e.g., 8
#plot_examples(test_images.numpy(), test_masks.numpy(), predictions_concatenated, num_examples=num_examples)

print(np.unique(predictions))
#pred_msk = np.where(predictions>0.5,1,0)
#print(np.unique(pred_msk))


def plot_examples(images, true_masks, predicted_masks, num_examples=23):
    plt.figure(figsize=(10, num_examples * 3))
    for i in range(num_examples):
        plt.subplot(num_examples, 3, i * 3 + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(num_examples, 3, i * 3 + 2)
        plt.imshow(true_masks[i].squeeze(), cmap='gray')
        plt.title("True Mask")
        plt.axis('off')

        plt.subplot(num_examples, 3, i * 3 + 3)
        plt.imshow(predicted_masks[i].squeeze(), cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Assuming predictions, test_images, and test_masks are ready
print(test_images.numpy().shape, test_masks.numpy().shape)
#plot_examples(test_images.numpy(), test_masks.numpy(), predictions_concatenated, num_examples=8)
#Now, you can visualize the original images, true masks, and predicted (thresholded) masks
num_examples_to_plot = min(len(test_images), 8)  # Choose a number that you have in your batch, e.g., 8
plot_examples(test_images.numpy(), test_masks.numpy(), predictions_concatenated, num_examples=num_examples_to_plot)