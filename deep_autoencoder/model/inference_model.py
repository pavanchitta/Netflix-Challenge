from model.base_model import BaseModel

class InferenceModel(BaseModel):

    def __init__(self, FLAGS):

        super(InferenceModel,self).__init__(FLAGS)
        self._init_parameters()
