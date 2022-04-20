from typing import Dict
import parse
import os
from pathlib import Path
import shutil
import argparse



def get_mapping_files(mapping_dir:str):
    all_mapping_files:Dict[str,str] = {}
    for root, dirs, files in os.walk(mapping_dir):
            for file in files:
                if file in all_mapping_files: raise Exception(f"{file} exists in 2 places:\n{all_mapping_files[file]}\n{root}")
                all_mapping_files[file] = os.path.join(root, file) 
    
    return all_mapping_files


def copy_to_target(target_dir_base:str, target_rel_dir: str, target_filename:str, mapping_file: str):
    target_full_dir = os.path.join(target_dir_base, target_rel_dir)
    if not os.path.exists(target_full_dir): os.makedirs(target_full_dir)
    
    target_file_path = os.path.join(target_full_dir, target_filename)
    shutil.copyfile(mapping_file, target_file_path)


def find_target_file(target_file:str, mapping_files:Dict[str, str], mapping_dir)->str:
    if target_file not in mapping_files: raise Exception(f"Unable to find: {target_file} in {mapping_dir}")
    return mapping_files[target_file]


def mirror_dir(SOURCE_PATTEN:str, TARGET_PATTEN:str, DATA_PATTEN:str, SOURCE_DIR:str, MAPPING_DIR:str, TARGET_DIR:str, verbose:bool=False):
    mapping_files = get_mapping_files(MAPPING_DIR)


    source_p =parse.compile(SOURCE_PATTEN)
    for root, dirs, files in os.walk(SOURCE_DIR):
        p = Path(root)
        rel_path = p.relative_to(SOURCE_DIR)
        for file in files:
            # match to patten
            match = source_p.parse(file)
            if match is None: continue
            target_file = TARGET_PATTEN.format(**match.named) # type: ignore
            data_file = DATA_PATTEN.format(**match.named) # type: ignore
            
            mapping_file_path = find_target_file(data_file, mapping_files, MAPPING_DIR)
            
            # copy to target dir
            copy_to_target(TARGET_DIR, str(rel_path), target_file,  mapping_file_path)
            if verbose:
                print(f"Copying: {mapping_file_path} -> {os.path.join(str(rel_path), target_file)}")


if __name__=="__main__":
    DEFAULT_SOURCE_PATTEN = "output-{id}.png"
    DEFAULT_TARGET_PATTEN = "output-{id}.json"
    DEFAULT_DATA_PATTEN = "output-{id}.json"

    DEFAULT_SOURCE_DIR = "./source"
    DEFAULT_MAPPING_DIR = "./mapping"
    DEFAULT_TARGET_DIR = "./target"

    parser = argparse.ArgumentParser(description='Extract Human Patches')
    parser.add_argument('-s', '--source_pat', type=str,help='Source Patten', default=DEFAULT_SOURCE_PATTEN)
    parser.add_argument('-p', '--data_pat', type=str, help='Data Patten', default=DEFAULT_DATA_PATTEN)
    parser.add_argument('-t', '--target_pat', type=str, help='Target Patten', default=DEFAULT_TARGET_PATTEN)
    parser.add_argument('-i', '--source_dir', type=str, help='Source Directory', default=DEFAULT_SOURCE_DIR)
    parser.add_argument('-d', '--data_dir', type=str, help='Data Directory', default=DEFAULT_MAPPING_DIR)
    parser.add_argument('-o', '--target_dir', type=str, help='Target Directory', default=DEFAULT_TARGET_DIR)
    parser.add_argument('-v', '--verbose', action="store_true", default = False)
    args = parser.parse_args()
    assert(os.path.isdir(args.source_dir))
    assert(os.path.isdir(args.data_dir))
    assert(os.path.isdir(args.target_dir))
    
    print(f"Making: {args.target_dir}   look like  {args.source_dir}")
    print(f"File mapping: {args.source_pat}  ->  {args.target_pat}")
    print(f"Using data from: {args.data_dir}")
    print(f"Using data search patten: {args.data_pat}")
    print("Do you want to continue?")
    y = input()
    if y!="y": exit()

    mirror_dir(args.source_pat, args.target_pat, args.data_pat, args.source_dir, args.data_dir, args.target_dir, args.verbose)

