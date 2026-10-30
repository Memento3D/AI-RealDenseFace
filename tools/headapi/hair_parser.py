"""Pinned CPU hair-region parser used by the head-cap experiment."""
import numpy as np
from PIL import Image
from .service import sha

PARSER_SHA='0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f'


class HairParser:
    def __init__(self,path):
        if sha(path)!=PARSER_SHA:
            raise ValueError('Face parser pin mismatch')
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=4
        self.session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])

    def __call__(self,image):
        image=Image.fromarray(np.asarray(image,dtype=np.uint8)) if not isinstance(image,Image.Image) else image
        image=image.convert('RGB')
        array=np.asarray(image.resize((512,512),Image.Resampling.BILINEAR),np.float32)/255
        array=(array-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        array=np.ascontiguousarray(array.transpose(2,0,1)[None])
        logits=self.session.run(None,{self.session.get_inputs()[0].name:array})[0]
        if logits.shape!=(1,19,512,512) or not np.isfinite(logits).all():
            raise ValueError('Unexpected parser output')
        labels=logits[0].argmax(0).astype(np.uint8)
        return np.asarray(Image.fromarray(labels).resize(image.size,Image.Resampling.NEAREST))
