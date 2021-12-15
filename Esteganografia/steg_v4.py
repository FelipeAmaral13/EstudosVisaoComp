import qrcode
import numpy as np
from PIL import Image


# Criar o a mensagem do qrcode
def qr_text(texto:str)->Image.Image:
    return qrcode.make(texto)

msg_secret = qr_text('Teste')


def encode(img_real:str, img_qrcode:qrcode.image.pil.PilImage):

    img_oficial = Image.open(img_real).convert('L')

    # Trocar a alt e lar da imagem que vai ser inserida na imgem oficial
    secret = img_qrcode.resize(img_oficial.size)

    # Formatacao da imagem
    data_c = np.array(img_oficial)
    data_s = np.array(secret, dtype=np.uint8)


    # Colocar as informaçoes no bits menos signficativos
    res = data_c & ~1 | data_s

    new_img = Image.fromarray(res).convert("L")
    return new_img

new_img = encode('lena.jpg', msg_secret)

def decode(img_encode:Image.Image):

    data_s = np.array(img_encode)
    # Pegar as informacoes dos bits menos significativos
    data_s = data_s & 1
    new_img = Image.fromarray(data_s * np.uint(255))

    return new_img

img_qr = decode(new_img)


def bitplanes(im:Image.Image):

    data = np.array(im)
    out = []

    # Crar uma img para cada  plato de k bits 
    for k in range(7,-1,-1):
    # Extrair o Kth bit (de 0 a 7)
        res = data // 2**k & 1
        out.append(res*255)

    # empilhar a geracao de imagens
    b = np.hstack(out)
    return Image.fromarray(b)

bitplanes(new_img).show()