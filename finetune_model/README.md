# Fine-tuning "vietnamese-content-cls" model

This repository contains the source code used to fine-tune and evaluate the `vietnamese-content-cls` model on Hugging Face. The fine-tuned model has achieved an accuracy of approximately 85% and an F1 score of around 83%.

## Dataset

A portion of the dataset from [Binhvq News Corpus](https://github.com/binhvq/news-corpus)  was used for training and validation. Specifically, 3.3 million samples were used for training, and 1.1 million samples each were used for validation and testing. The distribution of samples across classes is uniform

## Classes

The model classifies content into the following 13 classes:
- Công nghệ (Technology)
- Đời sống (Lifestyle)
- Giải trí (Entertainment)
- Giáo dục (Education)
- Khoa học (Science)
- Kinh tế (Economics)
- Nhà đất (Real Estate)
- Pháp luật (Law)
- Thế giới (World)
- Thể thao (Sports)
- Văn hóa (Culture)
- Xã hội (Society)
- Xe cộ (Vehicles)

## Files

- `train.py`: This file is used to fine-tune the model on the training and validation sets.
- `test.py`: This file is used to evaluate the final model on the test set.

## Performance

The model's achieved performance metrics are as follows:
- Accuracy: ~85%
- F1 Score: ~83%

These results demonstrate the model's effectiveness in classifying Vietnamese content.

