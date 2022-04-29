import argparse
import os
from Helpers.video import VideoReader
import cv2
from tqdm import tqdm
import random
import shutil
from Helpers.other import create_dirs

if __name__=="__main__":
    print("-------- Video to Images -------")

    DEFAULT_BASE_DIR = "../Dataset/Train"
    DEFAULT_OUT_DIR = "../Dataset/Generated/Background"
    DEFAULT_TEST_SAMPLES = 250


    parser = argparse.ArgumentParser(description='Extract Human Patches')
    parser.add_argument('-i', '--input', type=str,help='Input Directory', default=DEFAULT_BASE_DIR)
    parser.add_argument('-o', '--output', type=str,help='Output Directory', default=DEFAULT_OUT_DIR)
    parser.add_argument('-', '--test', type=int,help='Test Samples', default=DEFAULT_TEST_SAMPLES)
    args = parser.parse_args()
    
    games_base = os.path.join(args.input, "Games")
    movie_base = os.path.join(args.input, "Movie")

    games_files =  [f for f in os.listdir(games_base) if os.path.isfile(os.path.join(games_base, f))]
    movies_files =  [f for f in os.listdir(movie_base) if os.path.isfile(os.path.join(movie_base, f))]

    print(games_files)
    print(movies_files)

    games_our_dir = os.path.join(args.output, "A")
    movie_our_dir = os.path.join(args.output, "B")
    games_test_dir = os.path.join(args.output, "A_test")
    movie_test_dir = os.path.join(args.output, "B_test")
    create_dirs([games_our_dir, movie_our_dir, games_test_dir, movie_test_dir])
    
    print("Games")
    for file in games_files:
        print(f"\t{file}")
        file_no_extension = file.split(".")[0]
        vr = VideoReader(os.path.join(games_base, file))
        for i, frame in tqdm(enumerate(vr), total=len(vr)):
            cv2.imwrite(os.path.join(games_our_dir, f"{file_no_extension}-frame-{i}.jpg"), frame)
    

    print("Movies")
    for file in movies_files:
        print(f"\t{file}")
        file_no_extension = file.split(".")[0]
        vr = VideoReader(os.path.join(movie_base, file))
        for i, frame in tqdm(enumerate(vr), total=len(vr)):
            if i < 2*30 or i>len(vr)-2*30: continue # skip first and last 2 seconds 
            cv2.imwrite(os.path.join(movie_our_dir, f"{file_no_extension}-frame-{i}.jpg"), frame)

    
    game_frames =  [f for f in os.listdir(games_our_dir) if os.path.isfile(os.path.join(games_our_dir, f))]
    movie_frames =  [f for f in os.listdir(movie_our_dir) if os.path.isfile(os.path.join(movie_our_dir, f))]

    game_test_sample = random.sample(game_frames, args.test)
    movie_test_sample = random.sample(movie_frames, args.test)

    for gf, mf in zip(game_test_sample, movie_test_sample):
        shutil.move(os.path.join(games_our_dir, gf), os.path.join(games_test_dir, gf))
        shutil.move(os.path.join(movie_our_dir, mf), os.path.join(movie_test_dir, mf))
