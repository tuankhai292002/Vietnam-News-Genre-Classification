from pydantic import BaseModel,Field

class ClassifyRequest(BaseModel) :
    text: str = Field(example="Xe Massda CX-05 có giá bao nhiêu tiền ?")
    return_all_probabilities: bool = Field(True)
    enable_profanity_check: bool = Field(True)

class ClassifyResponse(BaseModel):
    result: dict = Field(
        example={
            "Xe cộ": 0.9861,
            "Công nghệ": 0.00598,
            "Nhà đất": 0.00358,
            "Xã hội": 0.00095,
            "Kinh tế": 0.00089,
            "Pháp luật": 0.00059,
            "Thế giới": 0.00039,
            "Khoa học": 0.00037,
            "Giải trí": 0.00035,
            "Thể thao": 0.00028,
            "Đời sống": 0.00027,
            "Giáo dục": 0.00013,
            "Văn hóa": 0.00013
        }
    )
    message: str = Field(example="Success")
    
