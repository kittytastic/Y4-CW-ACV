import os
import argparse
import sys
from tqdm import tqdm

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../"))

from Helpers.video_loader import VideoReader
from Helpers.image_dir import ImageDirWriter



if __name__=="__main__":
    print("-------- Video to Images -------")

    DEFAULT_BASE_DIR = "../Dataset/Train"
    DEFAULT_OUT_DIR = "../Dataset/Train_Raw"


    parser = argparse.ArgumentParser(description='Extract Human Patches')
    parser.add_argument('-i', '--input', type=str,help='Input Directory', default=DEFAULT_BASE_DIR)
    parser.add_argument('-o', '--output', type=str,help='Output Directory', default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    
    games_base = os.path.join(args.input, "Games")
    movie_base = os.path.join(args.input, "Movie")

    games_files =  [f for f in os.listdir(games_base) if os.path.isfile(os.path.join(games_base, f))]
    movies_files =  [f for f in os.listdir(movie_base) if os.path.isfile(os.path.join(movie_base, f))]

    print(games_files)
    print(movies_files)

    games_our_dir = os.path.join(args.output, "Games")
    movie_our_dir = os.path.join(args.output, "Movie")

    games_outstream = ImageDirWriter(games_our_dir, "game", "jpg")
    movie_outstream = ImageDirWriter(movie_our_dir, "movie", "jpg")

    print("Games")
    for file in games_files:
        print(f"\t{file}")
        vr = VideoReader(os.path.join(games_base, file))
        for frame in tqdm(vr):
            games_outstream.write_frame(frame)
    

    print("Movies")
    for file in movies_files:
        print(f"\t{file}")
        vr = VideoReader(os.path.join(movie_base, file))
        for frame in tqdm(vr):
            movie_outstream.write_frame(frame)
