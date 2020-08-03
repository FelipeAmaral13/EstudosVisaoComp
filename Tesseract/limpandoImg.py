from PIL import Image
import pytesseract

def limpaImg(img1, newImg):
    img = Image.open(img1)

    #Threshold para a imagem, e salva-la
    img = img.point(lambda x: 0 if x<100 else 255)
    img.save(newImg)

    return img

image = limpaImg('teste_tesseract2.png', 'ImgLimpa.png')

#Tesseract - OCR. Ler o que esta escrito na nova imagem
print(pytesseract.image_to_string(image))

