import os
import yaml
import torch
import nibabel as nib
from monai.apps import extractall
from pathlib import Path
import random
import shutil
import SimpleITK as sitk
# from torchio.transforms import ZNormalization
from PTDataSet import TorchDataSet
from parallel_sync import wget
from monai.networks.net import UNet


def win_scale(data, wl, ww, dtype, out_range):
    """
    Scale pixel intensity data using specified window level, width, and intensity range.
    """

    data_new = np.empty(data.shape, dtype=np.double)
    data_new.fill(out_range[1] - 1)

    data_new[data <= (wl - ww / 2.0)] = out_range[0]
    data_new[(data > (wl - ww / 2.0)) & (data <= (wl + ww / 2.0))] = \
        ((data[(data > (wl - ww / 2.0)) & (data <= (wl + ww / 2.0))] - (wl - 0.5)) / (ww - 1.0) + 0.5) * (out_range[1] - out_range[0]) + out_range[0]
    data_new[data > (wl + ww / 2.0)] = out_range[1] - 1

    return data_new.astype(dtype)


def preprocessing_ct(image):
    """Preprocess the CT images."""
    image = win_scale(image, 60, 400, np.float32, (0, 1))
    return image


def save_pt(image, name, save_dir, mask=None):
    """Save the images and labels as pytorch files. If without mask it is considered to be the test image and will be stored without a mask."""

    # checking if train with mask or test without mask.
    if mask is None:
        image = torch.from_numpy(image)

        #path = save_dir + "/" + str(name) + ".pt"
        path = os.path.join(save_dir, str(name) + ".pt")
        
        torch.save({"vol": image, "id": name}, path)
    else:
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        #path = save_dir + "/" + str(name) + ".pt"
        path = os.path.join(save_dir, str(name) + ".pt")
        
        torch.save({"vol": image, "mask": mask, "id": name}, path)


def convert_images(cfg, data_dir, save_dir):
    """Convert images to pytorch files."""

    # create lists of the images.
    label_list = os.listdir(os.path.join(data_dir, "labelsTr"))
    images_list = os.listdir(os.path.join(data_dir, "imagesTr"))
    image_list_test = os.listdir(os.path.join(data_dir, "imagesTs"))

    # remove the data that starts with . from the lists
    label_list = [x for x in label_list if not x.startswith(".")]
    images_list = [x for x in images_list if not x.startswith(".")]
    image_list_test = [x for x in image_list_test if not x.startswith(".")]

    # do the preprocessing based on the category.
    ct_list = ["Task03_Liver"]
    ct_preprocessing_decision = False
    if data_dir in ct_list:
        ct_preprocessing_decision = True

    # start the conversion of training images and training labels.
    # load the volumes from images and labels.
    for image, label in zip(images_list, label_list):
        # load the images and labels.
        image_path = os.path.join(data_dir, "imagesTr", image)
        label_path = os.path.join(data_dir, "labelsTr", label)

        # load the images and labels.
        image = nib.load(image_path)
        label = nib.load(label_path)

        image = image.get_fdata()
        label = label.get_fdata()
        # check for the datasets with more than 3 dimensions
        if len(image.shape) > 3:
            image = image.transpose(3, 2, 0, 1)
            label = label.transpose(2, 0, 1)

        else:
            # get image data and transpose to fit ZXY.
            image = image.transpose(2, 0, 1)

            label = label.transpose(2, 0, 1)

        # extract the name of the image.
        name = image_path.split("/")[-1].split(".")[0]

        # choose prepocessing based on the category.
        if ct_preprocessing_decision:
            image = preprocessing_ct(image)
        else:
            image = preprocessing_mr(image)

        # save the images and labels as pytorch files.
        save_pt(image, name, save_dir + "/train", label)

    # save the images in the new file structure

    # convert the test images.
    for image in image_list_test:
        # load the images and labels.
        image_path = os.path.join(data_dir, "imagesTs", image)

        # extract the name of the image.
        name = image_path.split("/")[-1].split(".")[0]

        # load the images and labels.
        image = nib.load(image_path)
        image = image.get_fdata()

        if len(image.shape) > 2:
            image = image.transpose(3, 2, 0, 1)

        else:
            # get image data and transpose to fit ZXY.
            image = image.transpose(2, 0, 1)

        # choose prepocessing based on the category.
        if ct_preprocessing_decision:
            image = preprocessing_ct(image)
        else:
            image = preprocessing_mr(image)

        # save the images and labels as pytorch files.
        save_pt(image, name, save_dir + "/test")


def prepare_conversion(cfg):
    """Collect the folders for the creation of the PyTorch files."""

    root_dir = cfg["data_storage"]["data_location"]
    pt_dir = cfg["data_storage"]["pt_location"]

    # create the root_dir path in the file system.
    Path(root_dir).mkdir(parents=False, exist_ok=True)
    Path(pt_dir).mkdir(parents=False, exist_ok=True)

    # retrive folder names.
    brain_dir, colon_dir, heart_dir, hippo_dir, liver_dir, lung_dir, pancreas_dir, prostate_dir, spleen_dir, vessel_dir = get_folders(root_dir)

    # create new folders from names for the tasks
    folder_list = ["Task03_Liver"]

    # create new folders from names in folder list in pt_dir
    for folder in folder_list:
        Path(os.path.join(pt_dir, folder)).mkdir(parents=False, exist_ok=True)
        # create a new folder for training and one for testing.
        Path(os.path.join(pt_dir, folder, "train")).mkdir(parents=False, exist_ok=True)
        Path(os.path.join(pt_dir, folder, "test")).mkdir(parents=False, exist_ok=True)

    # start the conversion to pt files.
    convert_images(cfg, liver_dir, os.path.join(pt_dir, "Task03_Liver"))



def get_folders(root_dir):
    """Return the folder locations."""
    
    liver_dir = os.path.join(root_dir, "Task03_Liver")
    return liver_dir


def train_test_split(cfg):
    """Split the training data randomly in train and validation based on a   70%/30% split."""
    root_dir = cfg["data_storage"]["pt_location"]
    liver_dir = get_folders(root_dir)

    # get in every folder, randomly select 30% and move them to an extra validation folder. (move, not copy)
    for f in [liver_dir]:
        # create new val folder.
        Path(os.path.join(f, "validation")).mkdir(parents=False, exist_ok=True)
        # get the list of files in the folder.
        file_list = os.listdir(os.path.join(f, "train", ))
        # get the number of files in the folder.
        file_number = len(file_list)
        # get the number of files to move.
        move_number = int(file_number * 0.3)
        # get the list of files to move.
        move_list = random.sample(file_list, move_number)
        # move the files to the validation folder.
        for file in move_list:
            shutil.move(os.path.join(f, "train", file), os.path.join(f, "validation"))


def download(root_dir, cfg):
    """Download the data from AWS Open Data Repository."""
    get_liver_aws = cfg["aws_links"]["liver"]


    # Liver
    compressed_file = os.path.join(root_dir, "Task03_Liver.tar")
    data_dir = os.path.join(root_dir, "Task03_Liver")
    if not os.path.exists(compressed_file):
        wget.download(root_dir, get_liver_aws)
        extractall(compressed_file, data_dir)

def main():
    # define the data directory
    with open("config.yml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    root_dir = cfg["data_storage"]["data_location"]

    # start by downloading the data
    download(root_dir, cfg)

    # then extract the data
    prepare_conversion(cfg)

    # start the train val split.
    train_test_split(cfg)

    # start the dataloading here. integrate the data augmentation.
    pt_path = cfg["data_storage"]["pt_location"]

    # FIXME: Search smaller dataset.
    pt_path_train = os.join.path(pt_path, "Task03_Liver", "train")
    pt_path_val = os.join.path(pt_path, "Task03_Liver", "validation")


    train = TorchDataSet(pt_path_train)
    val = TorchDataSet(pt_path_val)

    # create the dataloader.
    train_loader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val, batch_size=1, shuffle=False, num_workers=4)
    
    
    # initialize the model.
    model = UNet(spatial_dims=3, in_channels=1, out_channels=1)

    # initialize the optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # initialize the loss function.
    loss_function = torch.nn.CrossEntropyLoss()

    train_loss = 0
    val_loss = 0
    # easy training loop.
    for epoch in range(1, 100):
        # train the model.
        for x, y in train_loader:
            # forward pass.
            output = model(x)
            # calculate the loss.
            loss = loss_function(output, y)
            # backward pass.
            optimizer.zero_grad()
            loss.backward()
            # update the parameters.
            optimizer.step()
            # update the loss.
            train_loss += loss.item()


        # iterate over validation dataset.
        for x, y in val_loader:
            # forward pass.
            output = model(x)
            # calculate the loss.
            loss = loss_function(output, y)
            # combine loss
            val_loss += loss.item()

        # calculate the mean train_loss and val_loss.
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        # print the train_loss and val_loss.
        print("Epoch: {} \t Train Loss: {} \t Val Loss: {}".format(epoch, train_loss, val_loss))


    # save the model.
    torch.save(model.state_dict(), "model.pt")




if __name__ == "__main__":
    main()
