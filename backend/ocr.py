from detection import detection
from recognition import Recognition


def ocr(img, paddle_recog):
    _, _, line_imgs, _ = detection(img)
    output = paddle_recog.predict_batch_line_imgs(line_imgs)
    return "\n\n".join(output)


if __name__ == '__main__':
    paddle_recog = Recognition()
    img_path = "/home/love_you/Desktop/ocr_documents/35-3418a.png"
    result = ocr(img_path, paddle_recog)

    print(result)
