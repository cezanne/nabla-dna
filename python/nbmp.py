from PIL import Image

import os
import sys

import abmp


class NablaBitmap(abmp.AveragedBitmap):
    def __init__(self, dna_resolution=0, dna_depth=8, skip_nabla_sum=False):
        super().__init__(dna_resolution, dna_depth)
        self.dna_depth = dna_depth
        self.skip_nabla_sum = skip_nabla_sum

    def build_dna_bitmap(self, path, skip_normalization=False):
        self.build_averaged_bitmap(path)

        if not self.skip_nabla_sum:
            self.do_nabla_sum()
        else:
            self._vectorize()
        if not skip_normalization:
            self._normalize_gray()

        for i in range(len(self.bmp_dna)):
            self.bmp_dna[i] = int(round(self.bmp_dna[i] / 255 * (self.gray_depth - 1)))

    def _normalize_gray(self):
        v_max = v_min = self.bmp_dna[0]
        for i in range(len(self.bmp_dna)):
            if self.bmp_dna[i] < v_min:
                v_min = self.bmp_dna[i]
            if self.bmp_dna[i] > v_max:
                v_max = self.bmp_dna[i]
        if v_min == v_max:
            for i in range(len(self.bmp_dna)):
                self.bmp_dna[i] = 0
            return
        for i in range(len(self.bmp_dna)):
            self.bmp_dna[i] = (self.bmp_dna[i] - v_min) * 255 / (v_max - v_min)

    def do_nabla_sum(self):
        bmp_rot = []
        for h in range(int((self.dna_resolution + 1) / 2)):
            for w in range(h, self.dna_resolution - h):
                if (w >= self.dna_resolution - h - 1 and
                        (w != h or w != (self.dna_resolution + 1) / 2 - 1 or self.dna_resolution % 1 != 0)):
                    break
                self.bmp_dna[h][w] += self.bmp_dna[self.dna_resolution - 1 - w][h]                        # 90 degree
                self.bmp_dna[h][w] += self.bmp_dna[self.dna_resolution - 1 - h][self.dna_resolution - 1 - w]    # 180
                self.bmp_dna[h][w] += self.bmp_dna[w][self.dna_resolution - 1 - h]                        # -90
                self.bmp_dna[h][w] /= 4
                bmp_rot.append(self.bmp_dna[h][w])
        self.bmp_dna = bmp_rot

    def _vectorize(self):
        bmp_vec = []
        for h in range(self.dna_resolution):
            for w in range(self.dna_resolution):
                bmp_vec.append(self.bmp_dna[h][w])
        self.bmp_dna = bmp_vec

    def _save_nabla_bitmap(self, im):
        im.putalpha(0)
        i = 0
        for h in range(int((self.dna_resolution + 1) / 2)):
            for w in range(h, self.dna_resolution - h):
                if (w >= self.dna_resolution - h - 1 and
                        (w != h or w != (self.dna_resolution + 1) / 2 - 1 or self.dna_resolution % 1 != 0)):
                    break
                im.putpixel((w, h), (int(self.bmp_dna[i] * 256 / self.gray_depth), 255))
                i += 1

    def save_dna_bitmap(self, path):
        im = Image.new("LA", (self.dna_resolution, self.dna_resolution), 255)
        if not self.skip_nabla_sum:
            self._save_nabla_bitmap(im)
        im.save(path)

    def _write_dna_string(self, f, is_hex=False):
        for i in range(len(self.bmp_dna)):
            f.write(("%02x " if is_hex else "%d ") % self.bmp_dna[i])
        f.write('\n')

    def save_dna_text(self, path, is_hex=False):
        name_ext = os.path.splitext(path)
        path_out = name_ext[0] + f".x{self.dna_resolution:02d}d{self.dna_depth}" + name_ext[1]
        f = open(path_out, 'w')
        self._write_dna_string(f, is_hex)
        f.close()

    def show_dna_text(self):
        self._write_dna_string(sys.stdout, False)
