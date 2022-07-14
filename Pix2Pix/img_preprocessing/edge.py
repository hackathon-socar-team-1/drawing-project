import cv2
import glob

files = []
for i in range(1, 1042):
    file = glob.glob(f'./korea/all_lapl/a/a_{i}.png')
    files.extend(file)

i = 1
for f in files:
    gray_img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)

    laplacian_edge_img = cv2.Laplacian(gray_img, cv2.CV_8U, ksize=3)
    sobel_edge_img = cv2.Sobel(gray_img, cv2.CV_8U, 1, 0, 3)
    laplacian_edge_img = 255 - laplacian_edge_img
    sobel_edge_img = 255 - sobel_edge_img
    cv2.imwrite('./to_sketch/Laplacian.png', laplacian_edge_img)
    cv2.imwrite('./to_sketch/Sobel.png', sobel_edge_img)
    i += 1