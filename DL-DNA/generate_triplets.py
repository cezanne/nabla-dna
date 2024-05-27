import os

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

def filter_triplet_file(new_triplet_file):
    with open(new_triplet_file, 'r') as file:
        triplets = [line.strip().split() for line in file.readlines()]
    valid_triplets = [triplet for triplet in triplets if (triplet[2] != triplet[0] and triplet[2] != triplet[1]) and "(1)" not in triplet]
    

    with open(new_triplet_file, 'w') as file:
        for valid_triplet in valid_triplets:
            file.write(' '.join(valid_triplet) + '\n')

if __name__ == "__main__":
    images_dir = '../images'
    original_triplet_file = './triplet.txt'
    new_triplet_file = './triplet_new.txt'

    generate_new_triplet_file(images_dir, original_triplet_file, new_triplet_file)
    filter_triplet_file(new_triplet_file)