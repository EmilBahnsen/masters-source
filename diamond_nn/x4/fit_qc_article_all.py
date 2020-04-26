from diamond_nn.x4 import fit_qc_article
from diamond_nn.x4 import fit_qc_article_nc_qc_16neurons_relu
from diamond_nn.x4 import fit_qc_article_encoding_U

if __name__ == '__main__':
    for _ in range(10):
        for i in range(8):
            fit_qc_article_encoding_U.main(i)
