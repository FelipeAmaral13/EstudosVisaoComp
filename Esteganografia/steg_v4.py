import qrcode
import numpy as np
from PIL import Image
from pyzbar.pyzbar import decode

class QrCode:
    def __init__(self, message):
        self.img_qrcode = qrcode.make(message)
        self.img_encoded = None
    
    def encode(self, img_real):
        img_oficial = Image.open(img_real).convert('L')
        secret = self.img_qrcode.resize(img_oficial.size)
        data_c = np.array(img_oficial)
        data_s = np.array(secret, dtype=np.uint8)
        res = data_c & ~1 | data_s
        self.img_encoded = Image.fromarray(res).convert("L")
    
    def decode(self):
        data_s = np.array(self.img_encoded)
        data_s = data_s & 1
        img_qr = Image.fromarray(data_s * np.uint(255))
        return img_qr
    
    def get_text(self):
        result = decode(self.decode())
        print(result[0][0].decode('utf-8'))
    
    def plot_bitplanes(self):
        data = np.array(self.img_encoded)
        out = []

        for k in range(7, -1, -1):
            res = data // 2**k & 1
            out.append(res*255)

        b = np.hstack(out)
        Image.fromarray(b).show()



qrcode = QrCode('exemplo de mensagem')
qrcode.encode('lena.jpg')
qrcode.plot_bitplanes()
qrcode.get_text()
