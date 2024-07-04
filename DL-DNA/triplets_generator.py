import os
import argparse
import random

def generate_new_triplet_file(images_dir, original_triplet_file, new_triplet_file):
    with open(original_triplet_file, 'r') as file:
        triplets = [line.strip().split() for line in file.readlines()]

    all_images = set(os.listdir(images_dir))

    new_triplets = []
    for anchor, positive, _ in triplets:
        for new_negative in all_images:
            new_triplets.append((anchor, positive, new_negative))

    new_triplets.sort(key=lambda x: tuple(os.path.splitext(image)[0] for image in x))

    with open(new_triplet_file, 'w') as file:
        for triplet in new_triplets:
            file.write(' '.join(os.path.splitext(image)[0] for image in triplet) + '\n')

def filter_triplet_file(new_triplet_file, num_triplets):
    with open(new_triplet_file, 'r') as file:
        triplets = [line.strip().split() for line in file.readlines()]
    valid_triplets = [triplet for triplet in triplets if triplet[2] != triplet[0] and triplet[2] != triplet[1]]

    num_triplets = min(num_triplets, len(valid_triplets))
    selected_triplets = random.sample(valid_triplets, num_triplets)

    with open(new_triplet_file, 'w') as file:
        for valid_triplet in selected_triplets:
            file.write(' '.join(valid_triplet) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='new triplet 파일 생성')
    parser.add_argument('images_dir', type=str, help='이미지를 담은 파일')
    parser.add_argument('original_triplet_file', type=str, help='original triplet 파일 경로')
    parser.add_argument('new_triplet_file', type=str, help='new triplet 파일 경로')

    args = parser.parse_args()

    num_triplets = int(input("학습시키고 싶은 데이터의 개수를 입력하세요: "))

    generate_new_triplet_file(args.images_dir, args.original_triplet_file, args.new_triplet_file)
    filter_triplet_file(args.new_triplet_file, num_triplets)
