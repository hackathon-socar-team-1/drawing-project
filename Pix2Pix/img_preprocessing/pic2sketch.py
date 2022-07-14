import cv2

def img2sketch(photo, k_size, name):
    # Read Image
    img=cv2.imread(photo)
    
    # Convert to Gray Image
    gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert Image
    invert_img=cv2.bitwise_not(gray_img)

    # Blur image
    blur_img=cv2.GaussianBlur(invert_img, (k_size,k_size),0)

    # Invert Blurred Image
    invblur_img=cv2.bitwise_not(blur_img)

    # Sketch Image
    sketch_img=cv2.divide(gray_img,invblur_img, scale=256.0)

    # Save Sketch 
    cv2.imwrite(f'./to_sketch/{name}.png', sketch_img)


for i in range(1, 1042):
    img2sketch(photo=f'./a_{i}.png', k_size=7, name=f'a_{i}')