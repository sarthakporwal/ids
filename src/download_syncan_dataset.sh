#!/bin/bash


DEFAULT_DIRECTORY="../../datasets/can-ids/syncan/"
echo "Enter the directory to download the dataset"
echo "or just press Enter to accept following default directory:"
read -p "[${DEFAULT_DIRECTORY}]:" USER_DIRECTORY
DIRECTORY="${USER_DIRECTORY:-$DEFAULT_DIRECTORY}"


if [ -d "$DIRECTORY" ]; then
  echo "The folder $DIRECTORY already exists." 
  rm -rf "$DIRECTORY"
  echo "The existing files have been deleted!"
  echo "Downloading new files...."
else
  echo "The directory $DIRECTORY created." 
fi

git clone https://github.com/etas/SynCAN.git "$DIRECTORY"
echo "Raw SynCAN dataset downloaded in $DIRECTORY"
cd "$DEFAULT_DIRECTORY"
unzip 'train_*.zip' -d ambient
echo "Unzipped training dataset in datasets/can-ids/syncan/ambient"
unzip 'test_*.zip' -d attacks
echo "Unzipped training dataset in datasets/can-ids/syncan/attacks" 
rm -rf *.zip
rm -rf attacks/test_normal*
echo "SynCAN Data Downloaded!"


