import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class DCT:
    def __init__(self, N):
        self.N = N  # データ数．
        # 1次元，2次元離散コサイン変換の基底ベクトルをあらかじめ作っておく
        self.phi_1d = np.array([self.phi(i) for i in range(self.N)])

        # Nが大きいとメモリリークを起こすので注意
        # MNISTの28x28程度なら問題ない
        self.phi_2d = np.zeros((N, N, N, N))
        for i in range(N):
            for j in range(N):
                phi_i, phi_j = np.meshgrid(self.phi_1d[i], self.phi_1d[j])
                self.phi_2d[i, j] = phi_i * phi_j

    def dct(self, data):
        """1次元離散コサイン変換を行う"""
        return self.phi_1d.dot(data)

    def idct(self, c):
        """1次元離散コサイン逆変換を行う"""
        return np.sum(self.phi_1d.T * c, axis=1)

    def dct2(self, data):
        """2次元離散コサイン変換を行う"""
        return np.sum(self.phi_2d.reshape(N * N, N * N) * data.reshape(N * N), axis=1).reshape(N, N)

    def idct2(self, c):
        """2次元離散コサイン逆変換を行う"""
        return np.sum((c.reshape(N, N, 1) * self.phi_2d.reshape(N, N, N * N)).reshape(N * N, N * N),axis=0,).reshape(N, N)

    def phi(self, k):
        """離散コサイン変換(DCT)の基底関数"""
        # DCT-II
        if k == 0:
            return np.ones(self.N) / np.sqrt(self.N)
        else:
            return np.sqrt(2.0 / self.N) * np.cos((k * np.pi / (2 * self.N)) * (np.arange(self.N) * 2 + 1))
        # DCT-IV(試しに実装してみた)
        # return np.sqrt(2.0/N)*np.cos((np.pi*(k+0.5)/self.N)*(np.arange(self.N)+0.5))


def df(data, alpha, beta, bS=4):
    l, h, _ = data.shape
    for i in range(bS - 1, l - (bS - 1), 2):
        for j in range(h):
            p3 = data[j, i - 3]
            p2 = data[j, i - 2]
            p1 = data[j, i - 1]
            p0 = data[j, i]
            q0 = data[j, i + 1]
            q1 = data[j, i + 2]
            q2 = data[j, i + 3]
            q3 = data[j, i + 4]
            # print(p3, p2, p1, p0, q0, q1, q2, q3)
            if (abs(p2[0] - p0[0]) < beta) and (abs(p0[0] - q0[0]) < np.round(alpha / 4)):
                p0t = (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) / 8
                p1t = (p2 + p1 + p0 + q0 + 2) / 4
                p2t = (2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) / 8

            else:
                p0t = (2 * p1 + p0 + q1 + 2) / 4
                p1t, p2t = p1, p2

            if (abs(q2[0] - q0[0]) < beta) and (abs(p0[0] - q0[0]) < np.round(alpha / 4)):
                q0t = (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) / 8
                q1t = (p0 + q0 + q1 + q2 + 2) / 4
                q2t = (2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) / 8
            else:
                q0t = (2 * q1 + q0 + p1 + 2) / 4
                q1t, q2t = q1, q2

            data[j, i - 2] = p2t
            data[j, i - 1] = p1t
            data[j, i] = p0t
            data[j, i + 1] = q0t
            data[j, i + 2] = q1t
            data[j, i + 3] = q2t

    return data


def mse(original, converted):
    l, h, c = original.shape
    return np.sum(np.square(original.reshape(l * h * c) - converted.reshape(l * h * c))) / (l * h * c)


if __name__ == "__main__":
    N = 8
    dct = DCT(N)  # 離散コサイン変換を行うクラスを作成

    im = np.array(Image.open("lena.bmp"))
    im_c = np.zeros(im.shape)
    im_y = np.zeros(im.shape)

    # 8x8のパッチごとにdct変換を実行
    for i in range(im_c.shape[0] // N):
        for j in range(im_c.shape[1] // N):
            for k in range(im.shape[2]):
                im_c[N * i : N * (i + 1), N * j : N * (j + 1), k] = dct.dct2(im[N * i : N * (i + 1), N * j : N * (j + 1), k])

    # パラメータQによる量子化 q = {5, 10, 20, 40}
    q = 10
    print(im_c[0:5, 0:5, 1])
    im_c = np.round(im_c / q) * q
    print(im_c[0:5, 0:5, 1])

    # 8x8のパッチごとにdct逆変換を実行
    for i in range(im_c.shape[0] // N):
        for j in range(im_c.shape[1] // N):
            for k in range(im.shape[2]):
                im_y[N * i : N * (i + 1), N * j : N * (j + 1), k] = dct.idct2(im_c[N * i : N * (i + 1), N * j : N * (j + 1), k])

    # パラメータα, βによるDe-blocking Filter
    alpha = 20
    beta = 20
    im_y = df(im_y, alpha=alpha, beta=beta)

    # 元画像と変換画像間のMSEを計算する
    im_mse = mse(im, im_y.astype(int))
    print(f"MSE: {im_mse}")

    # 元の画像と復元したものを表示
    fig = plt.figure(figsize=(8, 4))
    axes = fig.subplots(1, 2)

    axes[0].imshow(im)
    axes[1].imshow(im_y.astype(int))

    axes[0].set_title("original")
    axes[1].set_title("restored")

    plt.suptitle(f"q={q}, alpha={alpha}, beta={beta}, mse={im_mse:.2f}")
    plt.tight_layout()
    plt.show()
