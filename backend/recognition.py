from paddleocr import PaddleOCR


class Recognition:
    def __init__(self, lang='en'):
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)

    def predict_line_img(self, img):
        ocr_result = self.ocr.ocr(img, det=False, cls=True)
        output = []
        for idx in range(len(ocr_result)):
            res = ocr_result[idx]
            for line in res:
                output.append(line[0])
        return output

    def predict_batch_line_imgs(self, batch_imgs):
        output = []
        for img in batch_imgs:
            output.extend(self.predict_line_img(img))
        return output



