DIR=$1
echo $DIR
cp -r Solution $DIR/
rm -r $DIR/Solution/pytorch_CycleGAN_and_pix2pix
cp -r Checkpoints $DIR/
cp environment.yml $DIR
cp README.md $DIR

mkdir $DIR/Results/
cp Dataset/Test/Video1/output/compare.mp4 $DIR/Results/side_by_side.mp4
cp Dataset/Test/Video1/output/final.mp4 $DIR/Results/full_screen.mp4