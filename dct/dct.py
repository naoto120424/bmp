import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class DCT:
    def __init__(self, N):
        self.N = N
        self.phi_1d = np.array([self.phi(i) for i in range(self.N)])

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
        N = self.N
        return np.sum(self.phi_2d.reshape(N * N, N * N) * data.reshape(N * N), axis=1).reshape(N, N)

    def idct2(self, c):
        """2次元離散コサイン逆変換を行う"""
        N = self.N
        return np.sum((c.reshape(N, N, 1) * self.phi_2d.reshape(N, N, N * N)).reshape(N * N, N * N),axis=0,).reshape(N, N)

    def phi(self, k):
        """離散コサイン変換(DCT)の基底関数"""
        if k == 0:
            return np.ones(self.N) / np.sqrt(self.N)
        else:
            return np.sqrt(2.0 / self.N) * np.cos((k * np.pi / (2 * self.N)) * (np.arange(self.N) * 2 + 1))


def df(data, alpha, beta, bS=4):
    l, h, c = data.shape
    for i in range(l):
        for j in range(bS - 1, h - bS):
            for k in range(c):
                p3 = data[i, j - 3, k]
                p2 = data[i, j - 2, k]
                p1 = data[i, j - 1, k]
                p0 = data[i, j, k]
                q0 = data[i, j + 1, k]
                q1 = data[i, j + 2, k]
                q2 = data[i, j + 3, k]
                q3 = data[i, j + 4, k]
                # print(p3, p2, p1, p0, q0, q1, q2, q3)
                if (abs(p2 - p0) < beta) and (abs(p0 - q0) < np.round(alpha / 4)):
                    p0t = (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) / 8
                    p1t = (p2 + p1 + p0 + q0 + 2) / 4
                    p2t = (2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) / 8
                else:
                    p0t = (2 * p1 + p0 + q1 + 2) / 4
                    p1t, p2t = p1, p2

                if (abs(q2 - q0) < beta) and (abs(p0 - q0) < np.round(alpha / 4)):
                    q0t = (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) / 8
                    q1t = (p0 + q0 + q1 + q2 + 2) / 4
                    q2t = (2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) / 8
                else:
                    q0t = (2 * q1 + q0 + p1 + 2) / 4
                    q1t, q2t = q1, q2

                data[i, j - 2, k] = p2t
                data[i, j - 1, k] = p1t
                data[i, j, k] = p0t
                data[i, j + 1, k] = q0t
                data[i, j + 2, k] = q1t
                data[i, j + 3, k] = q2t

    return data


def mse(original, converted):
    l, h, c = original.shape
    return np.sum(np.square(original.reshape(l * h * c) - converted.reshape(l * h * c))) / (l * h * c)


def main(q=10, alpha=20, beta=20):
    N = 8
    dct = DCT(N)

    # im = np.array(Image.open("lena.bmp"))
    im = np.array(Image.open("my_photo.png"))
    im_c = np.zeros(im.shape)
    im_y = np.zeros(im.shape)

    # 8x8のパッチごとにdct変換を実行
    for i in range(im_c.shape[0] // N):
        for j in range(im_c.shape[1] // N):
            for k in range(im.shape[2]):
                im_c[N * i : N * (i + 1), N * j : N * (j + 1), k] = dct.dct2(im[N * i : N * (i + 1), N * j : N * (j + 1), k])

    # パラメータQによる量子化 q = {5, 10, 20, 40}
    im_c = np.round(im_c / q) * q

    # 8x8のパッチごとにdct逆変換を実行
    for i in range(im_c.shape[0] // N):
        for j in range(im_c.shape[1] // N):
            for k in range(im.shape[2]):
                im_y[N * i : N * (i + 1), N * j : N * (j + 1), k] = dct.idct2(im_c[N * i : N * (i + 1), N * j : N * (j + 1), k])

    # パラメータα, βによるDe-blocking Filter
    im_y_df = im_y.copy()
    im_y_df = df(im_y_df, alpha=alpha, beta=beta)

    # 元画像と変換画像間のMSEを計算する
    im_y_mse = mse(im, im_y.astype(int))
    im_y_df_mse = mse(im, im_y_df.astype(int))
    print(f"Q: {q}, alpha: {alpha}, beta: {beta}")
    print(f"  Quantized : {im_y_mse}")
    print(f"  De-Blocked: {im_y_df_mse}")

    # 元の画像と復元したものを表示、保存
    result_path = "my_res"
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(os.path.join(result_path, f"q{q}"), exist_ok=True)

    fig = plt.figure(figsize=(15, 4))
    axes = fig.subplots(1, 3)

    axes[0].imshow(im)
    axes[1].imshow(im_y.astype(int))
    axes[2].imshow(im_y_df.astype(int))

    axes[0].set_title("original")
    axes[1].set_title(f"compressed\nmse={im_y_mse:.3f}")
    axes[2].set_title(f"de-blocked\nmse={im_y_df_mse:.3f}")

    plt.suptitle(f"Q={q}, alpha={alpha}, beta={beta}")
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, f"q{q}", f"alpha{alpha}_beta{beta}.jpg"), dpi=500)
    plt.close()


if __name__ == "__main__":
    q_list = [5, 10, 20, 40]
    alpha_list = [5, 10, 15, 20, 40]
    beta_list = [5, 10, 15, 20, 40]
    for q in q_list:
        for alpha in alpha_list:
            for beta in beta_list:
                main(q=q, alpha=alpha, beta=beta)
