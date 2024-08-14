import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import dl_dna_model
from sklearn.metrics.pairwise import cosine_similarity
from lineEnumerator import LineEnumerator

# 데이터 증강 설정
datagen = ImageDataGenerator(
    rotation_range=180,      # 0도에서 180도 사이로 무작위 회전
    width_shift_range=0.2,   # 이미지 너비의 20%까지 무작위로 수평 이동
    height_shift_range=0.2,  # 이미지 높이의 20%까지 무작위로 수직 이동
    shear_range=0.2,         # 이미지를 기울임
    zoom_range=0.2,          # 이미지 확대/축소 범위
    horizontal_flip=True,   #수직 반전
    vertical_flip=True,    # 수평 반전
    fill_mode='nearest'      # 빈 공간을 처리
)

def _triplet_loss(y_true, y_pred, alpha=2.0):
    anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
    return loss

class StackTripletEmbeddings(Layer):
    def call(self, inputs):
        anchor, positive, negative = inputs
        return tf.stack([anchor, positive, negative], axis=1)

class ModelTriplet(dl_dna_model.DlDnaModel):
    def __init__(self):
        input_shape = (224, 224, 3)
        base_model = MobileNet(weights='imagenet', include_top=False, pooling='avg', input_shape=input_shape)

        anchor_input = Input(shape=input_shape, name="anchor_input")
        positive_input = Input(shape=input_shape, name="positive_input")
        negative_input = Input(shape=input_shape, name="negative_input")

        anchor_embedding = Dense(128, activation='relu')(base_model(anchor_input))
        positive_embedding = Dense(128, activation='relu')(base_model(positive_input))
        negative_embedding = Dense(128, activation='relu')(base_model(negative_input))

        stacked_embeddings = StackTripletEmbeddings()([anchor_embedding, positive_embedding, negative_embedding])

        self.dl_model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=stacked_embeddings)
        self.dl_model.compile(optimizer=Adam(learning_rate=0.00005), loss=_triplet_loss)
        self.verbose_level = 1 if dl_dna_model.verbose else 0

    def _load_and_augment_triplet(self, triplet):
        images = []
        for name in triplet:
            image = dl_dna_model.load_img_data(name)
            image = image.reshape((1,) + image.shape)  # (1, 224, 224, 3)으로 reshape
            augmented_images = next(datagen.flow(image, batch_size=1))  # 증강된 이미지 생성
            images.append(augmented_images[0])  # 증강된 이미지를 리스트에 추가
        return np.array(images)

    def _train_triplet_model(self, triples):
        all_anchors, all_positives, all_negatives = [], [], []

        for triple in triples:
            images = self._load_and_augment_triplet(triple)
            all_anchors.append(images[0])
            all_positives.append(images[1])
            all_negatives.append(images[2])

        all_anchors = np.array(all_anchors)
        all_positives = np.array(all_positives)
        all_negatives = np.array(all_negatives)
        dummy_y = np.ones((len(triples), 1))

        self.dl_model.fit(x=[all_anchors, all_positives, all_negatives], y=dummy_y, epochs=50, verbose=self.verbose_level)  # Epochs를 50으로 설정

    def train(self, fpath_train: str):
        triples = list(LineEnumerator(fpath_train, True))
        self._train_triplet_model(triples)

    def extract_dna(self, image_path):
        image = dl_dna_model.load_img_data(image_path).reshape((1, 224, 224, 3))
        dummy_positive = np.zeros_like(image)
        dummy_negative = np.zeros_like(image)
        return self.dl_model.predict([image, dummy_positive, dummy_negative], verbose=self.verbose_level)[0]

    def save(self, path_save: str):
        self.dl_model.save(path_save)

    def load(self, path_load: str):
        self.dl_model = load_model(path_load, custom_objects={'_triplet_loss': _triplet_loss, 'StackTripletEmbeddings': StackTripletEmbeddings})

    def show_similarity(self, image_path1, image_path2):
        dna1 = self.extract_dna(image_path1)
        dna2 = self.extract_dna(image_path2)
        similarity = cosine_similarity(dna1.reshape(1, -1), dna2.reshape(1, -1))
        similarity_percent = similarity[0][0] * 100

        if similarity_percent >= 90:
            print(f"유사도 (퍼센트): {similarity_percent:.2f}% - 유사이미지")
        else:
            print(f"유사도 (퍼센트): {similarity_percent:.2f}% - 확인불가")