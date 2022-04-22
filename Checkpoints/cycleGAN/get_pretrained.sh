FILE=$1

echo "Note: available models are apple2orange, orange2apple, summer2winter_yosemite, winter2summer_yosemite, horse2zebra, zebra2horse, monet2photo, style_monet, style_cezanne, style_ukiyoe, style_vangogh, sat2map, map2sat, cityscapes_photo2label, cityscapes_label2photo, facades_photo2label, facades_label2photo, iphone2dslr_flower"

echo "Specified [$FILE]"
MODEL_NAME="paired_photo_to_monet_pretrained"
mkdir -p ${MODEL_NAME}
FILE="style_monet"
URL=http://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/$FILE.pth
wget -N $URL -O ./${MODEL_NAME}/latest_net_G_A.pth
FILE="monet2photo"
URL=http://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/$FILE.pth
wget -N $URL -O ./${MODEL_NAME}/latest_net_G_B.pth


NEW_MODEL_NAME="single_photo_to_monet_pretrained"
mkdir -p ${NEW_MODEL_NAME}
cp ${MODEL_NAME}/latest_net_G_A.pth ${NEW_MODEL_NAME}/latest_net_G.pth


NEW_MODEL_NAME="single_monet_to_photo_pretrained"
mkdir -p ${NEW_MODEL_NAME}
cp ${MODEL_NAME}/latest_net_G_B.pth ${NEW_MODEL_NAME}/latest_net_G.pth

MODEL_NAME="single_photo_to_cezanne_pretrained"
mkdir -p ${MODEL_NAME}
FILE="style_cezanne"
URL=http://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/$FILE.pth
wget -N $URL -O ./${MODEL_NAME}/latest_net_G.pth